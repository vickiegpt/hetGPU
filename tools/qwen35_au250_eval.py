#!/usr/bin/env python3
"""Run one deterministic Qwen 3.5 TQ1 llama-server evaluation mode."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


PROFILES = {
    "one-token": {
        "request_count": 1,
        "max_active": 1,
        "tokens_per_request": 1,
        "measurements": 1,
        "warmups": 0,
    },
    "full": {
        "request_count": 64,
        "max_active": 32,
        "tokens_per_request": 32,
        "measurements": 3,
        "warmups": 1,
    },
}
WARMUPS = PROFILES["full"]["warmups"]
MEASUREMENTS = PROFILES["full"]["measurements"]
REQUEST_COUNT = PROFILES["full"]["request_count"]
MAX_ACTIVE_REQUESTS = PROFILES["full"]["max_active"]
PREDICT_TOKENS = PROFILES["full"]["tokens_per_request"]
PROMPT_TOKENS = 256
PROMPT_TOKENS_BY_PROFILE = {"one-token": 16, "full": PROMPT_TOKENS}
CONTEXT_TOKENS_PER_REQUEST = 512
SERVER_CONTEXT_TOKENS = MAX_ACTIVE_REQUESTS * CONTEXT_TOKENS_PER_REQUEST
SEMANTIC_PROMPT = "Reply with exactly OK and no other text."
SEMANTIC_PROMPTS_BY_PROFILE = {
    "one-token": "Reply OK.",
    "full": SEMANTIC_PROMPT,
}
MODEL_SIZE = 94_155_830_880
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"
EXPECTED_EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}
ATTENTION_MARKERS = (
    "attention",
    "attn",
    "flash",
    "softmax",
    "qkv",
    "query",
    "key",
    "value",
    "kq",
    "qk",
)


class EvaluationError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path, expected_size, expected_sha256, verification_path=None):
    stat = path.stat()
    if stat.st_size != expected_size:
        raise EvaluationError("model byte count mismatch")
    if verification_path is None:
        if sha256(path) != expected_sha256:
            raise EvaluationError("model SHA-256 mismatch")
        return
    try:
        verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read model verification record: {error}") from error
    required = {
        "path": str(path),
        "size": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": expected_sha256,
    }
    if verification != required:
        raise EvaluationError("model changed after the independently hashed preflight")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_stream_events(path, status, events, parser_error=None):
    if path is None:
        return
    payload = {
        "schema_version": 1,
        "status": status,
        "events": events,
    }
    if parser_error is not None:
        payload["parser_error"] = parser_error
    atomic_json(path, payload)


def post_json(base_url, route, body, timeout):
    request = urllib.request.Request(
        base_url + route,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{route} failed: {error}") from error


def wait_for_health(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise EvaluationError(f"llama-server exited during startup with {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                body = json.load(response)
            if body.get("status") in ("ok", "no slot available"):
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.25)
    raise EvaluationError(f"llama-server did not become healthy: {last_error}")


def stream_completion(base_url, body, timeout, events_path=None):
    request = urllib.request.Request(
        base_url + "/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    first_token = None
    text_parts = []
    token_ids = []
    final = None
    events = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    raise EvaluationError(f"unexpected completion stream line: {line[:120]}")
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                event = json.loads(payload)
                events.append(event)
                error_payload = event.get("error")
                if error_payload is not None:
                    if isinstance(error_payload, dict):
                        message = error_payload.get("message", json.dumps(error_payload, sort_keys=True))
                    else:
                        message = str(error_payload)
                    raise EvaluationError(f"completion stream error: {message}")
                content = event.get("content", "")
                tokens = event.get("tokens", [])
                is_prompt_progress = "prompt_progress" in event
                if not is_prompt_progress and (content or tokens):
                    if first_token is None:
                        first_token = time.monotonic()
                    if not isinstance(content, str) or not isinstance(tokens, list):
                        raise EvaluationError("completion event content/tokens have invalid types")
                    text_parts.append(content)
                    token_ids.extend(tokens)
                if event.get("stop") is True:
                    final = event
    except EvaluationError as error:
        atomic_stream_events(events_path, "error", events, str(error))
        raise
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError) as error:
        message = f"/completion failed: {error}"
        atomic_stream_events(events_path, "error", events, message)
        raise EvaluationError(message) from error
    finished = time.monotonic()
    if final is None or first_token is None:
        message = "completion stream omitted final response or generated tokens"
        atomic_stream_events(events_path, "error", events, message)
        raise EvaluationError(message)
    timings = final.get("timings")
    if not isinstance(timings, dict):
        message = "completion final response omitted timings"
        atomic_stream_events(events_path, "error", events, message)
        raise EvaluationError(message)
    result = {
        "text": "".join(text_parts),
        "token_ids": token_ids,
        "ttft_ms": (first_token - started) * 1000.0,
        "end_to_end_ms": (finished - started) * 1000.0,
        "timings": timings,
        "events": events,
        "tokens_predicted": final.get("tokens_predicted"),
        "tokens_evaluated": final.get("tokens_evaluated"),
        "id_slot": final.get("id_slot"),
    }
    atomic_stream_events(events_path, "pass", events)
    return result


def stream_completion_batch(base_url, body, batch_size, timeout, events_path=None):
    request = urllib.request.Request(
        base_url + "/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    states = [
        {"text": [], "token_ids": [], "first_token": None, "final": None, "finished": None}
        for _ in range(batch_size)
    ]
    events = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    raise EvaluationError(f"unexpected batched completion stream line: {line[:120]}")
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                event = json.loads(payload)
                events.append(event)
                error_payload = event.get("error")
                if error_payload is not None:
                    if isinstance(error_payload, dict):
                        message = error_payload.get("message", json.dumps(error_payload, sort_keys=True))
                    else:
                        message = str(error_payload)
                    raise EvaluationError(f"batched completion stream error: {message}")
                index = event.get("index")
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < batch_size:
                    raise EvaluationError("batched completion event has an invalid index")
                state = states[index]
                content = event.get("content", "")
                tokens = event.get("tokens", [])
                is_prompt_progress = "prompt_progress" in event
                if not is_prompt_progress and (content or tokens):
                    if state["first_token"] is None:
                        state["first_token"] = time.monotonic()
                    if not isinstance(content, str) or not isinstance(tokens, list):
                        raise EvaluationError("batched completion event content/tokens have invalid types")
                    state["text"].append(content)
                    state["token_ids"].extend(tokens)
                if event.get("stop") is True:
                    state["final"] = event
                    state["finished"] = time.monotonic()
    except EvaluationError as error:
        atomic_stream_events(events_path, "error", events, str(error))
        raise
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError) as error:
        message = f"batched /completion failed: {error}"
        atomic_stream_events(events_path, "error", events, message)
        raise EvaluationError(message) from error
    results = []
    for index, state in enumerate(states):
        final = state["final"]
        first_token = state["first_token"]
        finished = state["finished"]
        if final is None or first_token is None or finished is None:
            message = f"batched completion {index} omitted final response or generated tokens"
            atomic_stream_events(events_path, "error", events, message)
            raise EvaluationError(message)
        timings = final.get("timings")
        if not isinstance(timings, dict):
            message = f"batched completion {index} omitted timings"
            atomic_stream_events(events_path, "error", events, message)
            raise EvaluationError(message)
        id_slot = final.get("id_slot")
        if isinstance(id_slot, bool) or not isinstance(id_slot, int):
            message = f"batched completion {index} omitted its slot assignment"
            atomic_stream_events(events_path, "error", events, message)
            raise EvaluationError(message)
        results.append({
            "text": "".join(state["text"]),
            "token_ids": state["token_ids"],
            "ttft_ms": (first_token - started) * 1000.0,
            "end_to_end_ms": (finished - started) * 1000.0,
            "timings": timings,
            "tokens_predicted": final.get("tokens_predicted"),
            "tokens_evaluated": final.get("tokens_evaluated"),
            "id_slot": id_slot,
        })
    if sorted(result["id_slot"] for result in results) != list(range(batch_size)):
        message = "batched completion did not use every server slot exactly once"
        atomic_stream_events(events_path, "error", events, message)
        raise EvaluationError(message)
    atomic_stream_events(events_path, "pass", events)
    return results


def exact_prompt(base_url, seed_path, timeout, prompt_tokens=PROMPT_TOKENS):
    seed = Path(seed_path).read_text(encoding="utf-8")
    tokenized = post_json(base_url, "/tokenize", {"content": seed, "add_special": False}, timeout)
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or len(tokens) < prompt_tokens:
        raise EvaluationError(f"prompt seed cannot provide exactly {prompt_tokens} tokens")
    selected = tokens[:prompt_tokens]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
        raise EvaluationError("prompt token IDs are not integers")
    detokenized = post_json(base_url, "/detokenize", {"tokens": selected}, timeout)
    prompt_text = detokenized.get("content")
    if not isinstance(prompt_text, str):
        raise EvaluationError("detokenize response omitted content")
    retokenized = post_json(
        base_url,
        "/tokenize",
        {"content": prompt_text, "add_special": False},
        timeout,
    ).get("tokens")
    if retokenized != selected:
        raise EvaluationError(
            f"retokenized prompt does not exactly match the selected {prompt_tokens} token IDs"
        )
    return prompt_text, selected


def completion_request(prompt, tokens_per_request=PREDICT_TOKENS):
    return {
        "prompt": prompt,
        "n_predict": tokens_per_request,
        "temperature": 0.0,
        "seed": 42,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
    }


def erase_all_slots(base_url, timeout, max_active=MAX_ACTIVE_REQUESTS):
    erased = []
    for id_slot in range(max_active):
        response = post_json(base_url, f"/slots/{id_slot}?action=erase", {}, timeout)
        if response.get("id_slot") != id_slot:
            raise EvaluationError(f"slot erase response did not bind slot {id_slot}")
        n_erased = response.get("n_erased")
        if isinstance(n_erased, bool) or not isinstance(n_erased, int) or n_erased < 0:
            raise EvaluationError(f"slot erase response for slot {id_slot} is invalid")
        erased.append(n_erased)
    return erased


def _validate_completed_request(item, request_id, prompt_tokens, tokens_per_request):
    tokens = item.get("token_ids", [])
    if (
        item.get("tokens_predicted") != tokens_per_request
        or item.get("tokens_evaluated") != prompt_tokens
        or len(tokens) != tokens_per_request
        or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
    ):
        raise EvaluationError(
            f"continuous request {request_id} did not execute the fixed "
            f"{prompt_tokens}+{tokens_per_request} workload"
        )


def _slot_refill_requests(base_url, request_body, timeout, profile, batch_started,
                          events_dir, events_prefix):
    """One request chain per slot; only that chain's completion permits reuse."""
    max_active = profile["max_active"]
    lock = threading.Lock()
    stopped = threading.Event()
    active = 0
    events = []
    erases = []

    def record(kind, request_id, slot, delta):
        nonlocal active
        # Caller owns lock so admission and failure-stop have one ordering.
        active += delta
        events.append({"event": kind, "request_id": request_id, "id_slot": slot,
                       "active": active,
                       "elapsed_ms": (time.monotonic() - batch_started) * 1000.0})

    def run_slot(slot):
        completed = []
        try:
            for request_id in range(slot, profile["request_count"], max_active):
                with lock:
                    if stopped.is_set():
                        return completed
                    started = time.monotonic()
                    record("admit", request_id, slot, 1)
                body = dict(request_body, id_slot=slot)
                path = None
                if events_dir is not None and events_prefix is not None:
                    path = Path(events_dir) / f"{events_prefix}-request-{request_id}-stream-events.json"
                try:
                    result = stream_completion(base_url, body, timeout, path)
                    if type(result.get("id_slot")) is not int or result["id_slot"] != slot:
                        raise EvaluationError(f"continuous request {request_id} returned the wrong slot")
                    _validate_completed_request(result, request_id, len(body["prompt"]),
                                                profile["tokens_per_request"])
                except BaseException:
                    with lock:
                        stopped.set()
                        record("error", request_id, slot, -1)
                    raise
                finished = time.monotonic()
                with lock:
                    record("complete", request_id, slot, -1)
                item = {k: v for k, v in result.items() if k != "events"}
                item.update(request_id=request_id,
                            queue_ms=(started - batch_started) * 1000.0,
                            service_ms=(finished - started) * 1000.0,
                            completed_from_batch_start_ms=(finished - batch_started) * 1000.0)
                completed.append(item)
                if request_id + max_active < profile["request_count"] and not stopped.is_set():
                    response = post_json(base_url, f"/slots/{slot}?action=erase", {}, timeout)
                    if (type(response.get("id_slot")) is not int or response["id_slot"] != slot
                            or type(response.get("n_erased")) is not int or response["n_erased"] < 0):
                        raise EvaluationError(f"slot erase response for slot {slot} is invalid")
                    with lock:
                        erases.append({"id_slot": slot, "after_request_id": request_id,
                                       "n_erased": response["n_erased"],
                                       "elapsed_ms": (time.monotonic() - batch_started) * 1000.0})
            return completed
        except BaseException:
            with lock:
                stopped.set()
            raise

    requests = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_active) as pool:
            futures = [pool.submit(run_slot, slot)
                       for slot in range(min(max_active, profile["request_count"]))]
            for future in concurrent.futures.as_completed(futures):
                requests.extend(future.result())
    finally:
        # Also retain a failed admission timeline; it never constitutes a pass.
        if events_dir is not None and events_prefix is not None:
            atomic_json(Path(events_dir) / f"{events_prefix}-admission.json", {
                "scheduler": "slot-refill", "status": "error" if stopped.is_set() else "pass",
                "events": events, "slot_reuse_evidence": erases,
            })
    return requests, sorted(erases, key=lambda e: e["after_request_id"]), events


def run_continuous_batch(
    base_url,
    request_body,
    timeout,
    profile=None,
    events_dir=None,
    events_prefix=None,
    scheduler="waves",
):
    profile = dict(PROFILES["full"] if profile is None else profile)
    request_count = profile["request_count"]
    max_active = profile["max_active"]
    tokens_per_request = profile["tokens_per_request"]
    prompt_tokens = len(request_body["prompt"])
    if scheduler not in ("waves", "slot-refill"):
        raise EvaluationError("unknown request scheduler")
    if any(type(value) is not int or value <= 0
           for value in (request_count, max_active, tokens_per_request)) or max_active > 32:
        raise EvaluationError("invalid request count or active slot bound (maximum 32)")
    batch_started = time.monotonic()
    requests = []
    wave_slot_erase_evidence = []
    slot_reuse_evidence = []
    admission_events = []
    if scheduler == "slot-refill":
        requests, slot_reuse_evidence, admission_events = _slot_refill_requests(
            base_url, request_body, timeout, profile, batch_started, events_dir, events_prefix,
        )
    for wave_start in range(0, request_count if scheduler == "waves" else 0, max_active):
        wave_stop = min(wave_start + max_active, request_count)
        active = wave_stop - wave_start
        wave_started = time.monotonic()
        body = dict(request_body)
        body["prompt"] = [request_body["prompt"] for _ in range(active)]
        events_path = None
        if events_dir is not None and events_prefix is not None:
            wave_index = wave_start // max_active
            events_path = (
                Path(events_dir)
                / f"{events_prefix}-wave-{wave_index}-stream-events.json"
            )
        if events_path is None:
            results = stream_completion_batch(base_url, body, active, timeout)
        else:
            results = stream_completion_batch(
                base_url, body, active, timeout, events_path
            )
        for index, result in enumerate(results):
            request_id = wave_start + index
            item = dict(result)
            item.update({
                "request_id": request_id,
                "queue_ms": (wave_started - batch_started) * 1000.0,
                "service_ms": result["end_to_end_ms"],
                "completed_from_batch_start_ms": (
                    wave_started - batch_started
                ) * 1000.0 + result["end_to_end_ms"],
            })
            requests.append(item)
        if wave_stop < request_count:
            wave_slot_erase_evidence.append(erase_all_slots(base_url, timeout, max_active))
    batch_finished = time.monotonic()
    requests.sort(key=lambda item: item["request_id"])
    if [item["request_id"] for item in requests] != list(range(request_count)):
        raise EvaluationError("continuous batch request IDs are missing or duplicated")
    for item in requests:
        _validate_completed_request(item, item["request_id"], prompt_tokens, tokens_per_request)
    wall_seconds = batch_finished - batch_started
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise EvaluationError("continuous batch wall time is not positive and finite")
    generated_tokens = request_count * tokens_per_request
    throughput = generated_tokens / wall_seconds
    if not math.isfinite(throughput) or throughput <= 0:
        raise EvaluationError("continuous batch throughput is not positive and finite")
    result = {
        "request_count": request_count,
        "max_active": max_active,
        "wave_count": math.ceil(request_count / max_active) if scheduler == "waves" else 0,
        "scheduler": scheduler,
        "slot_reuse_evidence": slot_reuse_evidence,
        "admission_events": admission_events,
        "tokens_per_request": tokens_per_request,
        "generated_tokens": generated_tokens,
        "wall_seconds": wall_seconds,
        "wave_slot_erase_evidence": wave_slot_erase_evidence,
        "requests": requests,
    }
    if profile == PROFILES["full"]:
        result["aggregate_generated_tokens_per_second"] = throughput
    return result


def enforce_aggregate_target(measurements, target_tps=28.0):
    if len(measurements) != 3:
        raise EvaluationError("full run omitted three measurements")
    if not math.isfinite(target_tps) or target_tps <= 0:
        raise EvaluationError("aggregate TPS target must be positive and finite")
    max_wall_seconds = 2048.0 / target_tps
    tps = []
    for item in measurements:
        scheduler = item.get("scheduler", "waves")
        expected_waves = {"waves": 2, "slot-refill": 0}.get(scheduler)
        if (expected_waves is None or item.get("generated_tokens") != 2048
                or item.get("wave_count") != expected_waves):
            raise EvaluationError(
                "full run did not execute fixed 64x32 workload with a known scheduler"
            )
        value = item.get("generation_tokens_per_second")
        wall_seconds = item.get("measured_wall_seconds")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds)
            or wall_seconds <= 0
        ):
            raise EvaluationError("full run contains invalid aggregate timing data")
        if value < target_tps:
            raise EvaluationError("full run is below the aggregate TPS target")
        if wall_seconds > max_wall_seconds:
            raise EvaluationError("full run exceeded the aggregate wall-time budget")
        if not math.isclose(value, 2048.0 / wall_seconds, rel_tol=1e-9, abs_tol=1e-9):
            raise EvaluationError("aggregate TPS does not equal generated tokens / measured wall time")
        tps.append(value)
    return {
        "target_tps": target_tps,
        "max_wall_seconds": max_wall_seconds,
        "minimum_tps": min(tps),
        "median_tps": statistics.median(tps),
        "maximum_tps": max(tps),
        "measurement_tps": tps,
    }


def semantic_completion_request(prompt):
    request = completion_request(prompt)
    request["n_predict"] = 1
    return request


def hardware_probe_request(prompt):
    request = completion_request(prompt)
    request["n_predict"] = 2
    return request


def templated_semantic_prompt(base_url, timeout, instruction=SEMANTIC_PROMPT, max_tokens=None):
    response = post_json(
        base_url,
        "/apply-template",
        {
            "messages": [{"role": "user", "content": instruction}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    prompt = response.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise EvaluationError("apply-template omitted the semantic prompt")
    if max_tokens is not None:
        tokenized = post_json(
            base_url,
            "/tokenize",
            {"content": prompt, "add_special": False},
            timeout,
        ).get("tokens")
        if not isinstance(tokenized, list) or len(tokenized) > max_tokens:
            actual = "invalid" if not isinstance(tokenized, list) else len(tokenized)
            raise EvaluationError(
                f"templated semantic prompt has {actual} tokens; limit is {max_tokens}"
            )
    return prompt


def parse_health(text):
    lowered = text.lower()
    good = "level 0 : 0x0 (good)" in lowered or "firewall" in lowered and "good" in lowered
    fatal = [line.strip() for line in text.splitlines() if re.search(r"\bfatal\b", line, re.IGNORECASE)]
    return {"firewall": "GOOD" if good else "BAD", "fatal_errors": fatal}


def capture_health(args, proof_dir, phase):
    path = proof_dir / f"xbutil-{phase}.txt"
    if args.health_fixture:
        text = Path(args.health_fixture).read_text(encoding="utf-8", errors="replace")
        path.write_text(text, encoding="utf-8")
    else:
        command = [
            args.xbutil,
            "examine",
            "-d",
            args.fpga_bdf,
            "-r",
            "platform",
            "-r",
            "thermal",
            "-r",
            "electrical",
            "-r",
            "error",
            "-r",
            "firewall",
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        text = result.stdout
        path.write_text(text, encoding="utf-8")
        if result.returncode != 0:
            raise EvaluationError(f"xbutil health capture failed in {phase}: exit {result.returncode}")
    parsed = parse_health(text)
    if parsed["firewall"] != "GOOD" or parsed["fatal_errors"]:
        raise EvaluationError(f"FPGA health is not clean in {phase}")
    return parsed


def parse_placement(log_text):
    matches = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", log_text, re.IGNORECASE)
    if not matches:
        raise EvaluationError("server log did not report GPU layer placement")
    offloaded, total = map(int, matches[-1])
    if total <= 0 or offloaded != total:
        raise EvaluationError(f"full CUDA placement failed: offloaded {offloaded}/{total} layers")
    return {"all_layers_on_gpu": True, "cpu_layers": 0}


def parse_load_ms(log_text):
    matches = re.findall(r"load time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", log_text, re.IGNORECASE)
    if matches:
        value = float(matches[-1])
    else:
        timestamp = r"(\d+)\.(\d{2})\.(\d{3})\.(\d{3})"
        started = re.search(
            rf"^{timestamp}.*\bload_model:\s+loading model\b",
            log_text,
            re.IGNORECASE | re.MULTILINE,
        )
        finished = re.search(
            rf"^{timestamp}.*\bllama_server:\s+model loaded\b",
            log_text,
            re.IGNORECASE | re.MULTILINE,
        )
        if started is None or finished is None:
            raise EvaluationError("server log did not report model load time")

        def elapsed_ms(match):
            minutes, seconds, milliseconds, microseconds = map(int, match.groups())
            return (minutes * 60 + seconds) * 1000 + milliseconds + microseconds / 1000.0

        value = elapsed_ms(finished) - elapsed_ms(started)
    if not math.isfinite(value) or value < 0:
        raise EvaluationError("server reported invalid model load time")
    return value


def parse_sampled_ffn_comparison(log_text, mode):
    records = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("event") == "captured_layer_comparison":
            records.append(record)
    if mode == "cuda":
        if records:
            raise EvaluationError("CUDA-only mode unexpectedly emitted an XRT FFN comparison")
        return None
    if len(records) != 1:
        raise EvaluationError(
            f"{mode} mode requires exactly one pre-timed sampled FFN comparison, got {len(records)}"
        )
    record = records[0]
    comparison = record.get("comparison")
    if (
        record.get("backend") != "xrt"
        or record.get("comparison_status") != "pass"
        or record.get("launch_ordinal") != 0
        or not isinstance(comparison, dict)
        or comparison.get("status") != "pass"
        or comparison.get("reference_backend") != "scalar_iq1s"
    ):
        raise EvaluationError("sampled FFN comparison metadata is invalid")
    checked = comparison.get("checked_elements")
    reference = comparison.get("reference_outputs")
    actual = comparison.get("actual_outputs")
    if (
        not _is_integer(checked)
        or checked <= 0
        or not isinstance(reference, list)
        or not isinstance(actual, list)
        or len(reference) != checked
        or len(actual) != checked
    ):
        raise EvaluationError("sampled FFN comparison vectors are incomplete")
    try:
        atol = float(comparison["atol"])
        rtol = float(comparison["rtol"])
        reported_max_abs = float(comparison["max_absolute_error"])
        reported_max_rel = float(comparison["max_relative_error"])
        pairs = [(float(left), float(right)) for left, right in zip(reference, actual)]
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"sampled FFN comparison contains invalid numbers: {error}") from error
    if atol != 1.0e-4 or rtol != 1.0e-3:
        raise EvaluationError("sampled FFN comparison used the wrong tolerance")
    if not all(math.isfinite(value) for pair in pairs for value in pair):
        raise EvaluationError("sampled FFN comparison contains non-finite output")
    absolute_errors = [abs(actual_value - reference_value) for reference_value, actual_value in pairs]
    relative_errors = [
        error / abs(reference_value) if reference_value != 0.0 else error
        for error, (reference_value, _) in zip(absolute_errors, pairs)
    ]
    if any(
        error > atol + rtol * abs(reference_value)
        for error, (reference_value, _) in zip(absolute_errors, pairs)
    ):
        raise EvaluationError("sampled FFN output is outside atol=1e-4, rtol=1e-3")
    max_abs = max(absolute_errors)
    max_rel = max(relative_errors)
    if (
        not math.isclose(reported_max_abs, max_abs, rel_tol=0.0, abs_tol=1.0e-7)
        or not math.isclose(reported_max_rel, max_rel, rel_tol=0.0, abs_tol=1.0e-7)
    ):
        raise EvaluationError("sampled FFN comparison error summary does not match its vectors")
    return {**comparison, "phase": "pre_timed", "kernel": record.get("kernel")}


def parse_routing(mode, evidence_path, require_evidence):
    zero_xrt = {
        "submission_count": 0,
        "completion_count": 0,
        "per_cu_submissions": [0, 0, 0, 0],
        "per_cu_completions": [0, 0, 0, 0],
        "request_ids": [],
        "stall_codes": [],
        "raw_min": 0,
        "raw_max": 0,
    }
    if mode == "cuda":
        return {"eligible": 0, "handled": 0, "fallback": 0, "error": 0}, zero_xrt
    if not evidence_path.exists():
        if require_evidence:
            raise EvaluationError("hybrid run did not create TQ1 routing evidence")
        return {"eligible": 1, "handled": 1, "fallback": 0, "error": 0}, {
            **zero_xrt,
            "submission_count": 4,
            "completion_count": 4,
            "per_cu_submissions": [1, 1, 1, 1],
            "per_cu_completions": [1, 1, 1, 1],
            "request_ids": [1, 2, 3, 4],
            "stall_codes": [1, 1, 1, 1],
        }

    records = []
    for line_number, line in enumerate(evidence_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid TQ1 evidence line {line_number}: {error}") from error
    if not records:
        raise EvaluationError("hybrid TQ1 evidence is empty")
    routes = {
        "eligible": len(records),
        "handled": sum(record.get("route") == "handled" for record in records),
        "fallback": sum(record.get("route") in ("fallback", "not_handled") for record in records),
        "error": sum(record.get("route") == "error" for record in records),
    }
    submissions = completions = 0
    per_submit = [0, 0, 0, 0]
    per_complete = [0, 0, 0, 0]
    request_ids = []
    stalls = []
    raw_min = 16_385
    raw_max = -16_385
    for operation_index, record in enumerate(records):
        if record.get("route") != "handled" or not isinstance(record.get("evidence"), dict):
            continue
        evidence = record["evidence"]
        operation_submissions = evidence.get("submission_count")
        operation_completions = evidence.get("completion_count")
        operation_stalls = evidence.get("stall_codes")
        operation_submit = evidence.get("per_cu_submissions")
        operation_complete = evidence.get("per_cu_completions")
        if not isinstance(operation_submissions, int) or not isinstance(operation_completions, int):
            raise EvaluationError("TQ1 evidence omitted submission/completion counts")
        if not isinstance(operation_stalls, list) or len(operation_stalls) != operation_completions:
            raise EvaluationError("TQ1 evidence stall accounting is incomplete")
        if not isinstance(operation_submit, list) or not isinstance(operation_complete, list) or len(operation_submit) != 4 or len(operation_complete) != 4:
            raise EvaluationError("TQ1 evidence per-CU accounting is incomplete")
        submissions += operation_submissions
        completions += operation_completions
        per_submit = [left + right for left, right in zip(per_submit, operation_submit)]
        per_complete = [left + right for left, right in zip(per_complete, operation_complete)]
        stalls.extend(operation_stalls)
        request_ids.extend((operation_index + 1) * (1 << 32) + index + 1 for index in range(operation_completions))
        raw_min = min(raw_min, evidence.get("raw_min", -16_385))
        raw_max = max(raw_max, evidence.get("raw_max", 16_385))
    return routes, {
        "submission_count": submissions,
        "completion_count": completions,
        "per_cu_submissions": per_submit,
        "per_cu_completions": per_complete,
        "request_ids": request_ids,
        "stall_codes": stalls,
        "raw_min": raw_min,
        "raw_max": raw_max,
    }


def zero_xrt_evidence():
    return {
        "submission_count": 0,
        "completion_count": 0,
        "per_cu_submissions": [0, 0, 0, 0],
        "per_cu_completions": [0, 0, 0, 0],
        "request_ids": [],
        "stall_codes": [],
        "raw_min": 0,
        "raw_max": 0,
        "reference_checked_components": 0,
        "operation_count": 0,
        "resident_matrix_hits": 0,
        "resident_matrix_misses": 0,
        "resident_matrix_bytes_transferred": 0,
        "program_cache_hits": 0,
        "program_cache_misses": 0,
        "host_pack_hits": 0,
        "host_pack_misses": 0,
        "host_pack_bytes_built": 0,
        "physical_completions": [],
    }


def load_jsonl_records(path, label, required):
    path = Path(path)
    if not path.exists():
        if required:
            raise EvaluationError(f"{label} was not created: {path}")
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"cannot read {label} {path}: {error}") from error
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"invalid {label} line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise EvaluationError(f"{label} line {line_number} is not a JSON object")
        records.append(record)
    if required and not records:
        raise EvaluationError(f"{label} is empty")
    return records


def is_iq1s_matmul(kernel):
    name = str(kernel).lower()
    return (
        "ggml_type19" in name
        and "stream_k_fixup" not in name
        and ("mul_mat_q" in name or "mul_mat_vec_q" in name)
    )


def is_attention(kernel):
    name = str(kernel).lower()
    return any(marker in name for marker in ATTENTION_MARKERS)


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _four_nonnegative_integers(value, label):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(_is_integer(item) and item >= 0 for item in value)
    ):
        raise EvaluationError(f"IQ1_S {label} must contain four nonnegative integers")
    return value


PERSISTENT_TIMING_FIELDS = {
    "capture", "route_dma", "trace_build_or_cache", "activation_pack",
    "activation_sync", "ring_publish", "doorbell", "device_wait",
    "completion_sync", "result_copy", "reconstruct", "compare", "log",
    "phase_wall",
}
PERSISTENT_PHASE_FIELDS = {
    "schema_version", "kind", "transaction_id", "layer_id", "phase",
    "trace_mode", "session_generation", "program_sha256", "semantic_sha256",
    "commands_per_cu", "completions_per_cu", "weight_dma_bytes",
    "eligible_direct_routes", "comparison_sampled", "reference_backend",
    "checked_elements", "max_abs_error", "max_rel_error", "nonfinite",
    "comparison_status", "timing_us",
}


def _persistent_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def parse_persistent_iq1s_routing(route_records, phase_records, mode):
    if mode not in ("handwritten", "compiler"):
        raise EvaluationError("persistent IQ1_S evidence requires a hybrid trace mode")
    if not isinstance(route_records, list) or not all(isinstance(item, dict) for item in route_records):
        raise EvaluationError("persistent IQ1_S route evidence must be a list of objects")
    if not isinstance(phase_records, list) or not phase_records or not all(
        isinstance(item, dict) for item in phase_records
    ):
        raise EvaluationError("persistent IQ1_S ledger is empty or malformed")
    direct = [item for item in route_records if is_iq1s_matmul(item.get("kernel"))]
    if direct:
        raise EvaluationError("persistent IQ1_S run emitted a legacy eligible direct route")
    attention = sum(
        is_attention(item.get("kernel")) and item.get("route") == "gpu"
        for item in route_records
    )
    if attention <= 0:
        raise EvaluationError("persistent IQ1_S run observed no native GPU attention")

    per_cu_commands = [0, 0, 0, 0]
    per_cu_completions = [0, 0, 0, 0]
    sampled = []
    seen = set()
    transaction_phases = {}
    generation = None
    for index, record in enumerate(phase_records):
        label = f"persistent phase[{index}]"
        if set(record) != PERSISTENT_PHASE_FIELDS:
            raise EvaluationError(f"{label} fields differ from schema")
        if record["schema_version"] != 2 or record["kind"] != "iq1s_persistent_phase":
            raise EvaluationError(f"{label} schema identity is invalid")
        transaction = record["transaction_id"]
        layer = record["layer_id"]
        if not _is_integer(transaction) or transaction <= 0 or not _is_integer(layer) or not 0 <= layer < 60:
            raise EvaluationError(f"{label} transaction/layer identity is invalid")
        phase = record["phase"]
        if phase not in ("A", "B") or record["trace_mode"] != mode:
            raise EvaluationError(f"{label} phase or trace mode is invalid")
        identity = (transaction, phase)
        if identity in seen:
            raise EvaluationError("persistent IQ1_S ledger duplicates a transaction/phase")
        seen.add(identity)
        sequence = transaction_phases.setdefault(transaction, [])
        sequence.append(phase)
        if sequence not in (["A"], ["A", "B"]):
            raise EvaluationError("persistent IQ1_S phase order is not A then optional B")
        current_generation = record["session_generation"]
        if not _is_integer(current_generation) or current_generation <= 0:
            raise EvaluationError(f"{label} session generation is invalid")
        if generation is None:
            generation = current_generation
        elif generation != current_generation:
            raise EvaluationError("persistent IQ1_S ledger spans session generations")
        programs = record["program_sha256"]
        if not isinstance(programs, list) or len(programs) != 4 or not all(
            _persistent_sha256(value) for value in programs
        ) or not _persistent_sha256(record["semantic_sha256"]):
            raise EvaluationError(f"{label} trace hashes are invalid")
        commands = _four_nonnegative_integers(record["commands_per_cu"], "persistent commands")
        completions = _four_nonnegative_integers(record["completions_per_cu"], "persistent completions")
        if not all(commands) or commands != completions:
            raise EvaluationError(f"{label} did not complete nonzero work on all four CUs")
        if record["weight_dma_bytes"] != 0 or record["eligible_direct_routes"] != 0:
            raise EvaluationError(f"{label} used measured weight DMA or a direct route")
        if record["nonfinite"] != 0:
            raise EvaluationError(f"{label} contains a nonfinite output")
        checked = record["checked_elements"]
        if not _is_integer(checked) or checked <= 0:
            raise EvaluationError(f"{label} checked element count is invalid")
        try:
            max_abs = float(record["max_abs_error"])
            max_rel = float(record["max_rel_error"])
        except (TypeError, ValueError) as error:
            raise EvaluationError(f"{label} comparison errors are invalid") from error
        if not math.isfinite(max_abs) or not math.isfinite(max_rel) or max_abs < 0 or max_rel < 0:
            raise EvaluationError(f"{label} comparison errors are invalid")
        if record["comparison_sampled"] is True:
            if (
                record["reference_backend"] != "libggml_dequantize_row_iq1_s"
                or record["comparison_status"] != "pass"
                or max_abs > 1.0e-4
                or max_rel > 1.0e-3
            ):
                raise EvaluationError(f"{label} sampled libggml comparison did not pass")
            sampled.append(record)
        elif record["comparison_sampled"] is False:
            if (
                record["reference_backend"] is not None
                or record["comparison_status"] != "finite_only"
                or max_abs != 0.0
                or max_rel != 0.0
            ):
                raise EvaluationError(f"{label} finite-only validation is invalid")
        else:
            raise EvaluationError(f"{label} comparison_sampled must be boolean")
        timing = record["timing_us"]
        if not isinstance(timing, dict) or set(timing) != PERSISTENT_TIMING_FIELDS or not all(
            _is_integer(value) and value >= 0 for value in timing.values()
        ):
            raise EvaluationError(f"{label} timing breakdown is invalid")
        per_cu_commands = [left + right for left, right in zip(per_cu_commands, commands)]
        per_cu_completions = [left + right for left, right in zip(per_cu_completions, completions)]

    if len(sampled) != 1:
        raise EvaluationError("persistent IQ1_S ledger requires exactly one libggml sample")
    sample = sampled[0]
    xrt = zero_xrt_evidence()
    xrt.update({
        "submission_count": sum(per_cu_commands),
        "completion_count": sum(per_cu_completions),
        "per_cu_submissions": per_cu_commands,
        "per_cu_completions": per_cu_completions,
        "reference_checked_components": sample["checked_elements"],
        "operation_count": len(phase_records),
    })
    routes = {
        "eligible": len(phase_records),
        "handled": len(phase_records),
        "fallback": 0,
        "error": 0,
        "eligible_kernels": [
            f"layer{record['layer_id']}.phase{record['phase']}" for record in phase_records
        ],
    }
    comparison = {
        "status": "pass",
        "reference_backend": sample["reference_backend"],
        "checked_elements": sample["checked_elements"],
        "atol": 1.0e-4,
        "rtol": 1.0e-3,
        "max_absolute_error": float(sample["max_abs_error"]),
        "max_relative_error": float(sample["max_rel_error"]),
        "phase": "pre_timed",
        "kernel": "iq1s_layer_persistent",
    }
    return routes, xrt, attention, comparison


def parse_iq1s_routing(route_records, xrt_records):
    if not isinstance(route_records, list) or not all(
        isinstance(record, dict) for record in route_records
    ):
        raise EvaluationError("IQ1_S route evidence must be a list of objects")
    if not isinstance(xrt_records, list) or not all(
        isinstance(record, dict) for record in xrt_records
    ):
        raise EvaluationError("IQ1_S XRT evidence must be a list of objects")

    eligible = [record for record in route_records if is_iq1s_matmul(record.get("kernel"))]
    handled = [
        record
        for record in eligible
        if record.get("route") == "cxl_tmatmul"
        and record.get("backend") == "xrt"
        and record.get("xrt_enabled") is True
        and record.get("hardware_matmul_enabled") is True
    ]
    fallback = [
        record for record in eligible if record.get("route") in ("gpu", "fallback")
    ]
    errors = [record for record in eligible if record.get("route") in ("reject", "error")]
    routes = {
        "eligible": len(eligible),
        "handled": len(handled),
        "fallback": len(fallback),
        "error": len(errors),
        "eligible_kernels": [record.get("kernel") for record in eligible],
    }
    if not eligible:
        raise EvaluationError("no eligible IQ1_S MMQ/MMVQ route was observed")
    if len(handled) != len(eligible) or fallback or errors:
        raise EvaluationError(f"IQ1_S routing was not strict and complete: {routes}")
    if len(handled) != len(xrt_records):
        raise EvaluationError(
            "IQ1_S route/XRT operation counts differ: "
            f"{len(handled)} routes, {len(xrt_records)} completions"
        )

    aggregate = zero_xrt_evidence()
    global_request_ids = set()
    for operation_index, record in enumerate(xrt_records):
        if record.get("event") != "au250_xrt_iq1s_completed":
            raise EvaluationError(f"unexpected IQ1_S XRT event: {record.get('event')!r}")
        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            raise EvaluationError("IQ1_S XRT record omitted its evidence object")
        if evidence.get("backend") != "xrt" or evidence.get("comparison_status") != "pass":
            raise EvaluationError("IQ1_S XRT comparison did not pass")
        submissions = evidence.get("submission_count")
        completions = evidence.get("completion_count")
        if (
            not _is_integer(submissions)
            or submissions <= 0
            or not _is_integer(completions)
            or completions != submissions
        ):
            raise EvaluationError("IQ1_S XRT submission/completion counts are invalid")
        per_submit = _four_nonnegative_integers(
            evidence.get("per_cu_submissions"), "per-CU submissions"
        )
        per_complete = _four_nonnegative_integers(
            evidence.get("per_cu_completions"), "per-CU completions"
        )
        if sum(per_submit) != submissions or sum(per_complete) != completions:
            raise EvaluationError("IQ1_S XRT per-CU totals do not account for every request")
        if per_submit != per_complete:
            raise EvaluationError("IQ1_S XRT per-CU submissions and completions differ")

        local_ids = evidence.get("request_ids")
        if (
            not isinstance(local_ids, list)
            or len(local_ids) != completions
            or not all(_is_integer(value) and 0 <= value <= 0xFFFF_FFFF for value in local_ids)
            or len(set(local_ids)) != len(local_ids)
        ):
            raise EvaluationError("IQ1_S XRT physical request IDs are invalid or duplicated")
        namespaced_ids = [((operation_index + 1) << 32) | value for value in local_ids]
        if any(value in global_request_ids for value in namespaced_ids):
            raise EvaluationError("IQ1_S XRT aggregate request IDs are duplicated")
        global_request_ids.update(namespaced_ids)

        stalls = evidence.get("stall_codes")
        if (
            not isinstance(stalls, list)
            or len(stalls) != completions
            or not all(_is_integer(value) and value != 0 for value in stalls)
        ):
            raise EvaluationError("IQ1_S XRT STALL completion evidence is invalid")
        raw_min = evidence.get("raw_min")
        raw_max = evidence.get("raw_max")
        if (
            not _is_integer(raw_min)
            or not _is_integer(raw_max)
            or not (-4096 <= raw_min <= raw_max <= 4096)
        ):
            raise EvaluationError("IQ1_S XRT raw result is outside [-4096, 4096]")
        checked = evidence.get("reference_checked_components")
        if not _is_integer(checked) or checked <= 0:
            raise EvaluationError("IQ1_S XRT evidence lacks reference-checked components")
        counter_names = (
            "resident_matrix_hits",
            "resident_matrix_misses",
            "resident_matrix_bytes_transferred",
            "program_cache_hits",
            "program_cache_misses",
            "host_pack_hits",
            "host_pack_misses",
            "host_pack_bytes_built",
        )
        counters = {}
        for name in counter_names:
            value = evidence.get(name)
            if not _is_integer(value) or value < 0:
                raise EvaluationError(f"IQ1_S XRT {name} is invalid")
            counters[name] = value
        physical = evidence.get("physical_completions")
        if not isinstance(physical, list) or len(physical) != completions:
            raise EvaluationError("IQ1_S XRT physical completion evidence is incomplete")
        physical_ids = []
        for item in physical:
            if not isinstance(item, dict):
                raise EvaluationError("IQ1_S XRT physical completion is not an object")
            local_id = item.get("request_id")
            if not _is_integer(local_id) or local_id not in local_ids:
                raise EvaluationError("IQ1_S XRT physical completion request ID is invalid")
            copied = dict(item)
            copied["request_id"] = ((operation_index + 1) << 32) | local_id
            physical_ids.append(copied["request_id"])
            aggregate["physical_completions"].append(copied)
        if sorted(physical_ids) != sorted(namespaced_ids):
            raise EvaluationError("IQ1_S XRT physical completions do not match request IDs")

        aggregate["submission_count"] += submissions
        aggregate["completion_count"] += completions
        aggregate["per_cu_submissions"] = [
            left + right for left, right in zip(aggregate["per_cu_submissions"], per_submit)
        ]
        aggregate["per_cu_completions"] = [
            left + right for left, right in zip(aggregate["per_cu_completions"], per_complete)
        ]
        aggregate["request_ids"].extend(namespaced_ids)
        aggregate["stall_codes"].extend(stalls)
        aggregate["reference_checked_components"] += checked
        for name, value in counters.items():
            aggregate[name] += value
        if operation_index == 0:
            aggregate["raw_min"] = raw_min
            aggregate["raw_max"] = raw_max
        else:
            aggregate["raw_min"] = min(aggregate["raw_min"], raw_min)
            aggregate["raw_max"] = max(aggregate["raw_max"], raw_max)
    aggregate["operation_count"] = len(xrt_records)
    attention = sum(
        is_attention(record.get("kernel")) and record.get("route") == "gpu"
        for record in route_records
    )
    return routes, aggregate, attention


def load_model_audit(path, expected_model_sha256):
    path = Path(path)
    try:
        raw = path.read_bytes()
        audit = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read Qwen model tensor audit: {error}") from error
    if not isinstance(audit, dict):
        raise EvaluationError("Qwen model tensor audit is not a JSON object")
    required = {
        "schema_version": 1,
        "status": "pass",
        "model_sha256": expected_model_sha256,
        "architecture": "qwen35moe",
        "routed_expert_count": 180,
        "routed_expert_types": EXPECTED_EXPERT_TYPES,
        "tq1_0_total": 0,
        "non_expert_iq1s": [],
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise EvaluationError(
                f"Qwen model tensor audit {key} is {audit.get(key)!r}, expected {expected!r}"
            )
    return audit, hashlib.sha256(raw).hexdigest()


def stop_process(process):
    if process.poll() is not None:
        return process.returncode
    process.send_signal(signal.SIGTERM)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=10)


def run(args):
    profile = {"name": args.profile, **PROFILES[args.profile]}
    profile_values = PROFILES[args.profile]
    prompt_tokens = PROMPT_TOKENS_BY_PROFILE[args.profile]
    proof_dir = Path(args.proof_dir)
    proof_dir.mkdir(parents=True, exist_ok=False)
    model = Path(args.model)
    server = Path(args.server)
    verify_model(model, args.model_size, args.model_sha256, args.model_verification)
    if sha256(server) != args.binary_sha256:
        raise EvaluationError("llama-server SHA-256 mismatch")

    model_audit = None
    model_audit_sha256 = None
    route_evidence_path = None
    xrt_evidence_path = None
    persistent_ledger_path = None
    if args.evidence_kind == "iq1s":
        if not args.model_audit:
            raise EvaluationError("IQ1_S evaluation requires --model-audit")
        model_audit, model_audit_sha256 = load_model_audit(
            args.model_audit, args.model_sha256
        )
        route_evidence_path = Path(
            args.route_evidence
            or os.environ.get("HETGPU_BITNET_ROUTE_LOG", proof_dir / "routes.jsonl")
        )
        xrt_evidence_path = Path(
            args.xrt_evidence
            or os.environ.get("HETGPU_XRT_EXECUTION_LOG", proof_dir / "xrt.jsonl")
        )
        if args.persistent_ledger:
            if args.mode == "cuda":
                raise EvaluationError("CUDA-only evaluation cannot use a persistent ledger")
            persistent_ledger_path = Path(args.persistent_ledger)
            if persistent_ledger_path.name != "phase-ledger.jsonl":
                raise EvaluationError("persistent ledger must be named phase-ledger.jsonl")
            configured_ledger = os.environ.get("HETGPU_QWEN_IQ1S_PROOF_LEDGER")
            if configured_ledger and Path(configured_ledger).resolve() != persistent_ledger_path.resolve():
                raise EvaluationError("persistent ledger CLI path differs from runtime environment")
        expected_parent = proof_dir.resolve()
        evidence_paths = [
            (route_evidence_path, "route evidence"),
        ]
        if persistent_ledger_path is None:
            evidence_paths.append((xrt_evidence_path, "XRT evidence"))
        else:
            evidence_paths.append((persistent_ledger_path, "persistent ledger"))
        for path, label in evidence_paths:
            if path.resolve().parent != expected_parent:
                raise EvaluationError(f"IQ1_S {label} must be directly beneath {proof_dir}")
            path.unlink(missing_ok=True)

    before_health = capture_health(args, proof_dir, "before")
    slot_save_path = proof_dir / "slot-state"
    slot_save_path.mkdir()
    server_batch = 16 if args.profile == "one-token" else profile_values["max_active"]
    command = [
        str(server), "--model", str(model), "--ctx-size", str(profile_values["max_active"] * CONTEXT_TOKENS_PER_REQUEST), "--n-gpu-layers", "999",
        "--threads", str(args.threads), "--host", "127.0.0.1", "--port", str(args.port),
        "--seed", "42", "--parallel", str(profile_values["max_active"]), "--reasoning", "off", "--verbosity", "4",
        "--batch-size", str(server_batch), "--ubatch-size", str(server_batch),
        "--cache-ram", "0", "--flash-attn", "on", "--no-cache-prompt", "--slot-save-path", str(slot_save_path),
        "--no-warmup", "--no-webui",
    ]
    (proof_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    server_environment = os.environ.copy()
    if args.server_preload:
        server_environment["LD_PRELOAD"] = args.server_preload
    environment_record = {
        key: value
        for key, value in server_environment.items()
        if key.startswith("HETGPU_")
        or key in (
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_LAUNCH_BLOCKING",
            "CUDA_VISIBLE_DEVICES",
            "GGML_CUDA_DISABLE_GRAPHS",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        )
    }
    atomic_json(proof_dir / "environment.json", environment_record)
    stdout_path = proof_dir / "server.stdout.log"
    stderr_path = proof_dir / "server.stderr.log"
    tq1_evidence_path = Path(
        os.environ.get("HETGPU_TQ1_EVIDENCE_LOG", proof_dir / "tq1-evidence.jsonl")
    )
    if args.evidence_kind == "tq1":
        tq1_evidence_path.unlink(missing_ok=True)
    process = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True, env=server_environment)
            base_url = f"http://127.0.0.1:{args.port}"
            wait_for_health(base_url, process, args.startup_timeout)
            prompt_text, prompt_ids = exact_prompt(
                base_url, args.prompt_seed, args.request_timeout, prompt_tokens
            )
            (proof_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
            atomic_json(proof_dir / "prompt-token-ids.json", prompt_ids)

            semantic_prompt = templated_semantic_prompt(
                base_url,
                args.request_timeout,
                SEMANTIC_PROMPTS_BY_PROFILE[args.profile],
                prompt_tokens if args.profile == "one-token" else None,
            )
            semantic_result = stream_completion(
                base_url,
                semantic_completion_request(semantic_prompt),
                args.request_timeout,
                proof_dir / "semantic-stream-events.json",
            )
            semantic_text = semantic_result["text"].strip()
            if semantic_text != "OK":
                raise EvaluationError(f"semantic response was {semantic_text!r}, expected exact 'OK'")
            if len(semantic_result["token_ids"]) != 1:
                raise EvaluationError("semantic gate did not produce exactly one token ID")

            # The first sampled token consumes prompt-evaluation logits.  A
            # second token forces one genuine single-token decode/MMVQ graph.
            hardware_probe = stream_completion(
                base_url,
                hardware_probe_request(semantic_prompt),
                args.request_timeout,
                proof_dir / "hardware-probe-stream-events.json",
            )
            if len(hardware_probe["token_ids"]) != 2:
                raise EvaluationError("hardware probe did not produce exactly two token IDs")

            semantic_hardware_gate = None
            if args.evidence_kind == "iq1s":
                semantic_route_records = load_jsonl_records(
                    route_evidence_path,
                    "IQ1_S route evidence",
                    required=args.mode != "cuda",
                )
                semantic_xrt_records = [] if persistent_ledger_path is not None else load_jsonl_records(
                    xrt_evidence_path, "IQ1_S XRT evidence", required=args.mode != "cuda"
                )
                if args.mode == "cuda":
                    if semantic_route_records or semantic_xrt_records:
                        raise EvaluationError("CUDA-only semantic gate unexpectedly created IQ1_S evidence")
                else:
                    if persistent_ledger_path is None:
                        gate_routes, gate_xrt, gate_attention = parse_iq1s_routing(
                            semantic_route_records, semantic_xrt_records
                        )
                    else:
                        semantic_phases = load_jsonl_records(
                            persistent_ledger_path,
                            "persistent IQ1_S phase ledger",
                            required=True,
                        )
                        gate_routes, gate_xrt, gate_attention, _ = parse_persistent_iq1s_routing(
                            semantic_route_records, semantic_phases, args.mode
                        )
                    if gate_attention <= 0:
                        raise EvaluationError("IQ1_S semantic gate observed no native GPU attention")
                    if not all(value > 0 for value in gate_xrt["per_cu_completions"]):
                        raise EvaluationError("IQ1_S semantic gate did not activate all four CUs")
                    semantic_hardware_gate = {
                        "routes": gate_routes,
                        "xrt": gate_xrt,
                        "gpu_attention_routes": gate_attention,
                    }

            timed_request = completion_request(prompt_ids, profile_values["tokens_per_request"])
            warmups = []
            slot_erase_evidence = []
            for warmup_index in range(profile_values["warmups"]):
                slot_erase_evidence.append(
                    erase_all_slots(base_url, args.request_timeout, profile_values["max_active"])
                )
                warmups.append(
                    run_continuous_batch(
                        base_url,
                        timed_request,
                        args.request_timeout,
                        profile_values,
                        proof_dir,
                        f"warmup-{warmup_index}",
                        scheduler=args.scheduler,
                    )
                )
            measured = []
            for measurement_index in range(profile_values["measurements"]):
                slot_erase_evidence.append(
                    erase_all_slots(base_url, args.request_timeout, profile_values["max_active"])
                )
                measured.append(
                    run_continuous_batch(
                        base_url,
                        timed_request,
                        args.request_timeout,
                        profile_values,
                        proof_dir,
                        f"measurement-{measurement_index}",
                        scheduler=args.scheduler,
                    )
                )
            generated_by_request = [
                item["token_ids"] for item in measured[0]["requests"]
            ]
            for index, batch in enumerate(measured[1:], start=1):
                current = [item["token_ids"] for item in batch["requests"]]
                if current != generated_by_request:
                    raise EvaluationError(
                        f"measured continuous batch {index} generated different token IDs"
                    )
            atomic_json(proof_dir / "warmups.json", warmups)
            atomic_json(proof_dir / "measured-requests.json", measured)
    finally:
        server_exit = stop_process(process) if process is not None else None

    after_health = capture_health(args, proof_dir, "after")
    log_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    placement = parse_placement(log_text)
    load_ms = parse_load_ms(log_text)
    sampled_ffn_comparison = (
        None
        if persistent_ledger_path is not None
        else parse_sampled_ffn_comparison(log_text, args.mode)
    )
    if args.evidence_kind == "tq1":
        routes, xrt = parse_routing(
            args.mode, tq1_evidence_path, args.require_routing_evidence
        )
        gpu_attention_routes = None
    elif args.mode == "cuda":
        route_records = load_jsonl_records(
            route_evidence_path, "IQ1_S route evidence", required=False
        )
        xrt_records = load_jsonl_records(
            xrt_evidence_path, "IQ1_S XRT evidence", required=False
        )
        if route_records or xrt_records:
            raise EvaluationError("CUDA-only mode unexpectedly created IQ1_S route/XRT evidence")
        routes = {
            "eligible": 0,
            "handled": 0,
            "fallback": 0,
            "error": 0,
            "eligible_kernels": [],
        }
        xrt = zero_xrt_evidence()
        gpu_attention_routes = 0
    else:
        route_records = load_jsonl_records(
            route_evidence_path,
            "IQ1_S route evidence",
            required=args.require_routing_evidence,
        )
        if persistent_ledger_path is None:
            xrt_records = load_jsonl_records(
                xrt_evidence_path,
                "IQ1_S XRT evidence",
                required=args.require_routing_evidence,
            )
            routes, xrt, gpu_attention_routes = parse_iq1s_routing(
                route_records, xrt_records
            )
        else:
            phase_records = load_jsonl_records(
                persistent_ledger_path,
                "persistent IQ1_S phase ledger",
                required=True,
            )
            routes, xrt, gpu_attention_routes, sampled_ffn_comparison = (
                parse_persistent_iq1s_routing(route_records, phase_records, args.mode)
            )
    measurements = []
    for batch in measured:
        requests = batch["requests"]
        measurement = {
            "model_load_ms": load_ms,
            "prompt_tokens_per_second": statistics.median(
                item["timings"].get("prompt_per_second") for item in requests
            ),
            "ttft_ms": statistics.median(item["ttft_ms"] for item in requests),
            "single_request_generation_tokens_per_second": statistics.median(
                item["timings"].get("predicted_per_second") for item in requests
            ),
            "end_to_end_ms": statistics.median(
                item["end_to_end_ms"] for item in requests
            ),
            "queue_ms": statistics.median(item["queue_ms"] for item in requests),
            "service_ms": statistics.median(item["service_ms"] for item in requests),
            "measured_wall_seconds": batch["wall_seconds"],
            "request_count": batch["request_count"],
            "max_active": batch["max_active"],
            "wave_count": batch["wave_count"],
            "scheduler": batch["scheduler"],
            "generated_tokens": batch["generated_tokens"],
        }
        if args.profile == "full":
            measurement["generation_tokens_per_second"] = batch[
                "aggregate_generated_tokens_per_second"
            ]
        measurements.append(measurement)
    record = {
        "schema_version": 2 if args.evidence_kind == "iq1s" else 1,
        "evidence_kind": args.evidence_kind,
        "mode": args.mode,
        "profile": profile,
        "model_size": args.model_size,
        "model_sha256": args.model_sha256,
        "llama_revision": args.llama_revision,
        "binary_sha256": args.binary_sha256,
        "placement": placement,
        "prompt_tokens": prompt_tokens,
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_by_request[0],
        "generated_token_ids_by_request": generated_by_request,
        "token_equivalence_measurement": 0,
        "slot_ids_by_request": [item["id_slot"] for item in measured[0]["requests"]],
        "slot_erase_evidence": slot_erase_evidence,
        "wave_slot_erase_evidence": [
            batch["wave_slot_erase_evidence"] for batch in warmups + measured
        ],
        "scheduler": args.scheduler,
        "scheduler_evidence": [{
            "admission_events": batch["admission_events"],
            "slot_reuse_evidence": batch["slot_reuse_evidence"],
            "wall_seconds": batch["wall_seconds"],
        } for batch in warmups + measured],
        "semantic": {"text": semantic_text, "token_ids": semantic_result["token_ids"]},
        "hardware_probe": {"token_ids": hardware_probe["token_ids"], "n_predict": 2},
        "sampled_ffn_comparison": sampled_ffn_comparison,
        "semantic_hardware_gate": semantic_hardware_gate,
        "routes": routes,
        "xrt": xrt,
        "gpu_attention_routes": gpu_attention_routes,
        "model_audit_sha256": model_audit_sha256,
        "measurements": measurements,
        "process": {"exit_code": 0, "server_termination_code": server_exit},
        "device_health": {"before": before_health, "after": after_health},
        "request_contract": timed_request,
        "warmup_count": len(warmups),
        "request_count": profile_values["request_count"],
        "max_active_requests": profile_values["max_active"],
        "generated_tokens_per_request": profile_values["tokens_per_request"],
        "context_tokens_per_request": CONTEXT_TOKENS_PER_REQUEST,
        "server_context_tokens": profile_values["max_active"] * CONTEXT_TOKENS_PER_REQUEST,
    }
    atomic_json(proof_dir / f"{args.mode}.json", record)
    return record


def _report_metric_rows(cuda, hybrid):
    metric_specs = (
        ("Prompt tokens/s", "prompt_tokens_per_second"),
        ("Generation tokens/s", "generation_tokens_per_second"),
        ("Time to first token (ms)", "ttft_ms"),
        ("End-to-end latency (ms)", "end_to_end_ms"),
    )
    rows = []
    for label, key in metric_specs:
        try:
            cuda_metric = cuda[key]
            hybrid_metric = hybrid[key]
            values = [
                float(cuda_metric[field])
                for field in ("median", "min", "max", "population_stdev")
            ] + [
                float(hybrid_metric[field])
                for field in ("median", "min", "max", "population_stdev")
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"normalized proof has invalid {key}: {error}") from error
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise EvaluationError(f"normalized proof has non-finite or negative {key}")
        cuda_median, cuda_minimum, cuda_maximum, cuda_stdev = values[:4]
        hybrid_median, hybrid_minimum, hybrid_maximum, hybrid_stdev = values[4:]
        if cuda_median == 0.0:
            raise EvaluationError(f"normalized proof has zero CUDA median for {key}")
        ratio = hybrid_median / cuda_median
        rows.append(
            f"| {label} | {cuda_median:.6g} | {hybrid_median:.6g} | {ratio:.6g} | "
            f"{cuda_minimum:.6g}-{cuda_maximum:.6g} / {hybrid_minimum:.6g}-{hybrid_maximum:.6g} | "
            f"{cuda_stdev:.6g} / {hybrid_stdev:.6g} |"
        )
    return rows


def _iq1s_report_metric_rows(cuda, handwritten, compiler):
    metric_specs = (
        ("Prompt tokens/s", "prompt_tokens_per_second"),
        ("Generation tokens/s", "generation_tokens_per_second"),
        ("Time to first token (ms)", "ttft_ms"),
        ("End-to-end latency (ms)", "end_to_end_ms"),
    )
    rows = []
    for label, key in metric_specs:
        medians = []
        for mode_name, metrics in (
            ("cuda", cuda),
            ("handwritten", handwritten),
            ("compiler", compiler),
        ):
            try:
                metric = metrics[key]
                values = [
                    float(metric[field])
                    for field in ("median", "min", "max", "population_stdev")
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise EvaluationError(
                    f"normalized proof has invalid {mode_name} {key}: {error}"
                ) from error
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise EvaluationError(
                    f"normalized proof has non-finite or negative {mode_name} {key}"
                )
            medians.append(values[0])
        rows.append(
            f"| {label} | {medians[0]:.6g} | {medians[1]:.6g} | {medians[2]:.6g} |"
        )
    return rows


def render_iq1s_report(normalized, proof_path):
    if not isinstance(normalized, dict) or normalized.get("schema_version") != 3:
        raise EvaluationError("IQ1_S report generation requires a schema-3 normalized proof")
    if normalized.get("status") != "pass":
        raise EvaluationError("IQ1_S report generation requires a passing normalized proof")
    try:
        model = normalized["model"]
        audit = normalized["model_audit"]
        cuda_mode = normalized["modes"]["cuda"]
        handwritten_mode = normalized["modes"]["handwritten"]
        compiler_mode = normalized["modes"]["compiler"]
    except (KeyError, TypeError) as error:
        raise EvaluationError(f"normalized IQ1_S proof omitted report evidence: {error}") from error

    expected_model = {
        "size": MODEL_SIZE,
        "sha256": MODEL_SHA256,
        "architecture": "qwen35moe",
        "llama_revision": LLAMA_REVISION,
    }
    if not isinstance(model, dict) or any(model.get(key) != value for key, value in expected_model.items()):
        raise EvaluationError("normalized IQ1_S proof model identity is invalid")
    binary_sha256 = model.get("binary_sha256")
    if not isinstance(binary_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is None:
        raise EvaluationError("normalized IQ1_S proof binary identity is invalid")

    if not isinstance(audit, dict):
        raise EvaluationError("normalized IQ1_S proof model audit is invalid")
    if (
        audit.get("routed_expert_count") != 180
        or audit.get("routed_expert_types") != EXPECTED_EXPERT_TYPES
        or audit.get("tq1_0_total") != 0
        or audit.get("non_expert_iq1s") != []
    ):
        raise EvaluationError("normalized IQ1_S proof model audit does not match the mixed-format contract")
    expected_tensor_coverage = EXPECTED_EXPERT_TYPES["IQ1_S"] / 180
    tensor_coverage = normalized.get("tensor_eligibility_coverage")
    if (
        isinstance(tensor_coverage, bool)
        or not isinstance(tensor_coverage, (int, float))
        or not math.isfinite(float(tensor_coverage))
        or not math.isclose(float(tensor_coverage), expected_tensor_coverage, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise EvaluationError("normalized IQ1_S proof tensor eligibility coverage is invalid")

    if normalized.get("token_ids_match") is not True or normalized.get("all_cus_active") is not True:
        raise EvaluationError("normalized IQ1_S proof did not validate tokens and all four CUs")
    if normalized.get("eligible_route_coverage") != 1.0:
        raise EvaluationError("normalized IQ1_S proof route coverage is not 100%")
    xrt_by_mode = {}
    for mode_name, mode in (
        ("handwritten", handwritten_mode),
        ("compiler", compiler_mode),
    ):
        routes = mode.get("routes") if isinstance(mode, dict) else None
        xrt = mode.get("xrt") if isinstance(mode, dict) else None
        if not isinstance(routes, dict):
            raise EvaluationError(f"normalized IQ1_S proof {mode_name} routes are invalid")
        eligible = routes.get("eligible")
        if (
            isinstance(eligible, bool)
            or not isinstance(eligible, int)
            or eligible <= 0
            or routes.get("handled") != eligible
            or routes.get("fallback") != 0
            or routes.get("error") != 0
        ):
            raise EvaluationError(
                f"normalized IQ1_S proof {mode_name} routing is not strict and complete"
            )
        if not isinstance(xrt, dict):
            raise EvaluationError(f"normalized IQ1_S proof {mode_name} XRT evidence is invalid")
        per_cu = xrt.get("per_cu_completions")
        submission_count = xrt.get("submission_count")
        completion_count = xrt.get("completion_count")
        if (
            not isinstance(per_cu, list)
            or len(per_cu) != 4
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in per_cu
            )
            or not isinstance(submission_count, int)
            or isinstance(submission_count, bool)
            or submission_count <= 0
            or completion_count != submission_count
            or sum(per_cu) != completion_count
        ):
            raise EvaluationError(
                f"normalized IQ1_S proof {mode_name} XRT completion accounting is invalid"
            )
        xrt_by_mode[mode_name] = (per_cu, submission_count, completion_count)

    for mode_name, mode in (
        ("cuda", cuda_mode),
        ("handwritten", handwritten_mode),
        ("compiler", compiler_mode),
    ):
        if not isinstance(mode, dict) or mode.get("measurements") != MEASUREMENTS:
            raise EvaluationError(f"normalized IQ1_S proof has invalid {mode_name} measurement count")
    rows = _iq1s_report_metric_rows(
        cuda_mode.get("metrics", {}),
        handwritten_mode.get("metrics", {}),
        compiler_mode.get("metrics", {}),
    )

    if normalized.get("sampled_ffn_within_tolerance") is not True:
        raise EvaluationError("normalized IQ1_S proof did not validate sampled FFN output")
    numerical_errors = {}
    for mode_name, mode in (("handwritten", handwritten_mode), ("compiler", compiler_mode)):
        try:
            comparison = mode["sampled_ffn_comparison"]
            error = float(comparison["max_absolute_error"])
            relative_error = float(comparison["max_relative_error"])
        except (KeyError, TypeError, ValueError) as exception:
            raise EvaluationError(f"normalized IQ1_S proof has invalid {mode_name} FFN evidence") from exception
        if (
            not isinstance(comparison, dict)
            or comparison.get("status") != "pass"
            or comparison.get("reference_backend") != "scalar_iq1s"
            or comparison.get("atol") != 1.0e-4
            or comparison.get("rtol") != 1.0e-3
            or not math.isfinite(error)
            or error < 0.0
            or not math.isfinite(relative_error)
            or relative_error < 0.0
        ):
            raise EvaluationError(f"normalized IQ1_S proof has unqualified {mode_name} FFN evidence")
        numerical_errors[mode_name] = error

    return "\n".join(
        [
            "# Qwen3.5-397B-A17B Mixed-Quant CUDA vs AU250 IQ1_S Evaluation",
            "",
            "## Qualification result",
            "",
            "- Proof status: PASS",
            "- Model: Qwen3.5-397B-A17B-UD-TQ1_0.gguf",
            f"- Model SHA-256: {MODEL_SHA256}",
            f"- llama.cpp revision: {LLAMA_REVISION}",
            f"- Runtime binary SHA-256: {binary_sha256}",
            "- Tensor audit: 141 IQ1_S, 24 IQ2_XXS, 4 IQ3_S, and 11 MXFP4 routed-expert tensors",
            "- Tensor eligibility: 141/180 routed-expert tensors eligible",
            "- IQ2_XXS, IQ3_S, and MXFP4 remained on CUDA",
            "- Eligible IQ1_S operations handled by AU250: 100%",
            "- Token IDs identical: yes",
            "- Active CUs: 4/4 in handwritten and compiler modes",
            "",
            "## Method",
            "",
            "Both modes used the same verified GGUF and binary, full CUDA layer placement, "
            "a 512-token context, greedy seed-42 sampling, an exact 256-token timed prompt, "
            "and 32 generated tokens per request. Each mode used one 64-request warm-up and "
            "three measured 64-request continuous batches with at most 32 active requests.",
            "",
            "Hybrid intercepted only exact type-19 IQ1_S MMQ/MMVQ launches for the 141 eligible "
            "routed-expert tensors. CUDA retained attention, linear attention, routing, shared "
            "experts, normalization, KV/state work, sampling, and the other 39 routed-expert "
            "tensors. AU250 execution used assembler-produced 128-bit tmatmul instructions and "
            "the matrix/input/output/program four-BO ABI.",
            "",
            "## Performance",
            "",
            "| Metric | CUDA-only median | Handwritten median | Compiler median |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            "## Numerical and physical evidence",
            "",
            f"Sampled FFN maximum absolute error was "
            f"{numerical_errors['handwritten']:.6g} for handwritten traces and "
            f"{numerical_errors['compiler']:.6g} for compiler traces, with "
            "atol=1e-4 and rtol=1e-3.",
            "",
            f"Handwritten per-CU completions: `{xrt_by_mode['handwritten'][0]}`; compiler "
            f"per-CU completions: `{xrt_by_mode['compiler'][0]}`. The normalized proof binds unique "
            "request IDs, nonzero STALL codes, raw-output bounds, exact completion ownership, "
            f"and pre/post device health to `{Path(proof_path)}`.",
            "",
            "## Limitations",
            "",
            "This is aggregate continuous-batch text generation on one host and one selected mixed-format "
            "checkpoint. Only the 141 audited IQ1_S routed-expert tensors were eligible for "
            "AU250 execution. The evaluation does not measure vision, multi-user serving, "
            "speculative decoding, attention offload, or a CPU-expert baseline.",
            "",
        ]
    )


def render_report(normalized, proof_path):
    if not isinstance(normalized, dict) or normalized.get("status") != "pass":
        raise EvaluationError("report generation requires a passing normalized proof")
    try:
        cuda = normalized["modes"]["cuda"]["metrics"]
        hybrid_mode = normalized["modes"]["hybrid"]
        hybrid = hybrid_mode["metrics"]
        routes = hybrid_mode["routes"]
        xrt = hybrid_mode["xrt"]
    except (KeyError, TypeError) as error:
        raise EvaluationError(f"normalized proof omitted report evidence: {error}") from error
    if normalized.get("token_ids_match") is not True or normalized.get("all_cus_active") is not True:
        raise EvaluationError("normalized proof did not validate tokens and all four CUs")
    if normalized.get("eligible_route_coverage") != 1.0:
        raise EvaluationError("normalized proof route coverage is not 100%")
    if routes.get("eligible", 0) <= 0 or routes.get("handled") != routes.get("eligible") or routes.get("fallback") or routes.get("error"):
        raise EvaluationError("normalized proof routing is not strict and complete")
    per_cu = xrt.get("per_cu_completions")
    if not isinstance(per_cu, list) or len(per_cu) != 4 or not all(isinstance(value, int) and value > 0 for value in per_cu):
        raise EvaluationError("normalized proof does not contain four active CUs")

    rows = _report_metric_rows(cuda, hybrid)

    return "\n".join(
        [
            "# Qwen3.5-397B-A17B TQ1_0 CUDA vs AU250 Hybrid Evaluation",
            "",
            "## Qualification result",
            "",
            "- Proof status: PASS",
            "- Model: Qwen3.5-397B-A17B-UD-TQ1_0.gguf",
            f"- Model SHA-256: {MODEL_SHA256}",
            f"- llama.cpp revision: {LLAMA_REVISION}",
            "- Token IDs identical: yes",
            "- Eligible expert operations handled by AU250: 100%",
            f"- Active CUs: {sum(value > 0 for value in per_cu)}/4",
            "",
            "## Method",
            "",
            "Both modes used the same binary, fully CUDA-resident model, 512-token context, "
            "greedy sampling, exact 256-token timed prompt, and 32 generated tokens. Each "
            "mode used one warm-up and five measured requests in a fresh process while "
            "retaining the model across requests. Hybrid changed only strict TQ1_0 routed-"
            "expert `mul_mat_id` dispatch; attention, linear attention, routing, shared "
            "experts, normalization, KV/state work, and sampling remained on CUDA.",
            "",
            "## Performance",
            "",
            "| Metric | CUDA-only median | Hybrid median | Hybrid/CUDA | CUDA / hybrid min-max | CUDA / hybrid population stdev |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Offload evidence",
            "",
            f"Per-CU completions: `{per_cu}`. The validated proof, including raw-output "
            f"bounds, unique request IDs, STALL codes, and pre/post device telemetry, is `{Path(proof_path)}`.",
            "",
            "## Limitations",
            "",
            "This is batch-size-one text generation on one host and one selected checkpoint. "
            "It does not measure vision, multi-user serving, speculative decoding, attention "
            "offload, or a CPU-expert baseline.",
            "",
        ]
    )


def render_report_command(arguments):
    report_parser = argparse.ArgumentParser(prog="qwen35_au250_eval.py render-report")
    report_parser.add_argument("--normalized", required=True)
    report_parser.add_argument("--proof-path", required=True)
    report_parser.add_argument("--output", required=True)
    args = report_parser.parse_args(arguments)
    try:
        normalized = json.loads(Path(args.normalized).read_text(encoding="utf-8"))
        report = render_report(normalized, Path(args.proof_path))
        repository = Path(__file__).resolve().parents[1]
        evaluation_root = (repository / "zluda" / "evaluation").resolve()
        output = Path(args.output).resolve()
        if output.parent != evaluation_root:
            raise EvaluationError(f"report output must be directly beneath {evaluation_root}")
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(report, encoding="utf-8")
        os.replace(temporary, output)
    except (EvaluationError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"QWEN_TQ1_REPORT_FAILED: {error}", file=sys.stderr)
        return 1
    return 0


def render_iq1s_report_command(arguments):
    report_parser = argparse.ArgumentParser(prog="qwen35_au250_eval.py render-iq1s-report")
    report_parser.add_argument("--normalized", required=True)
    report_parser.add_argument("--proof-path", required=True)
    report_parser.add_argument("--output", required=True)
    args = report_parser.parse_args(arguments)
    try:
        normalized = json.loads(Path(args.normalized).read_text(encoding="utf-8"))
        report = render_iq1s_report(normalized, Path(args.proof_path))
        repository = Path(__file__).resolve().parents[1]
        evaluation_root = (repository / "zluda" / "evaluation").resolve()
        output = Path(args.output).resolve()
        if output.parent != evaluation_root:
            raise EvaluationError(f"report output must be directly beneath {evaluation_root}")
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(report, encoding="utf-8")
        os.replace(temporary, output)
    except (EvaluationError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"QWEN_IQ1S_REPORT_FAILED: {error}", file=sys.stderr)
        return 1
    return 0


def parser():
    result = argparse.ArgumentParser()
    result.add_argument(
        "--mode", choices=("cuda", "handwritten", "compiler"), required=True
    )
    result.add_argument("--profile", choices=tuple(PROFILES), required=True)
    result.add_argument("--scheduler", choices=("waves", "slot-refill"), default="waves")
    result.add_argument("--evidence-kind", choices=("tq1", "iq1s"), default="tq1")
    result.add_argument("--server", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--prompt-seed", required=True)
    result.add_argument("--proof-dir", required=True)
    result.add_argument("--port", required=True, type=int)
    result.add_argument("--threads", required=True, type=int)
    result.add_argument("--model-size", type=int, default=MODEL_SIZE)
    result.add_argument("--model-sha256", default=MODEL_SHA256)
    result.add_argument("--model-verification")
    result.add_argument("--llama-revision", default=LLAMA_REVISION)
    result.add_argument("--binary-sha256", required=True)
    result.add_argument("--server-preload")
    result.add_argument("--startup-timeout", type=float, default=1800.0)
    result.add_argument("--request-timeout", type=float, default=3600.0)
    result.add_argument("--xbutil", default="xbutil")
    result.add_argument("--fpga-bdf", default="0000:64:00.1")
    result.add_argument("--health-fixture")
    result.add_argument("--route-evidence")
    result.add_argument("--xrt-evidence")
    result.add_argument("--persistent-ledger")
    result.add_argument("--model-audit")
    result.add_argument("--require-routing-evidence", action="store_true")
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == "render-iq1s-report":
        return render_iq1s_report_command(arguments[1:])
    if arguments and arguments[0] == "render-report":
        return render_report_command(arguments[1:])
    args = parser().parse_args(arguments)
    try:
        result = run(args)
    except (EvaluationError, OSError, ValueError, TypeError) as error:
        prefix = "QWEN_IQ1S_EVALUATION_FAILED" if args.evidence_kind == "iq1s" else "QWEN_TQ1_EVALUATION_FAILED"
        print(f"{prefix}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"mode": result["mode"], "status": "pass"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
