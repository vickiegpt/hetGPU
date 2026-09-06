#!/usr/bin/env python3
import json
import hashlib
import math
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import importlib.util


EVALUATOR = Path(__file__).parents[2] / "tools" / "qwen35_au250_eval.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("qwen35_eval", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_continuous_batch_runs_64_requests_as_four_16_prompt_waves(monkeypatch):
    evaluator = load_evaluator()
    observed_batches = []

    def fake_stream_batch(_base_url, body, batch_size, _timeout):
        observed_batches.append(body)
        assert batch_size == 16
        return [{
            "token_ids": list(range(32)),
            "tokens_predicted": 32,
            "tokens_evaluated": 256,
            "ttft_ms": 1.0,
            "end_to_end_ms": 2.0,
            "timings": {"prompt_per_second": 200.0, "predicted_per_second": 10.0},
            "id_slot": index,
        } for index in range(batch_size)]

    monkeypatch.setattr(evaluator, "stream_completion_batch", fake_stream_batch)
    monkeypatch.setattr(evaluator, "erase_all_slots", lambda *_args: [287] * 16)
    batch = evaluator.run_continuous_batch(
        "http://127.0.0.1:1",
        evaluator.completion_request(list(range(256))),
        timeout=10,
    )

    assert len(batch["requests"]) == 64
    assert [item["request_id"] for item in batch["requests"]] == list(range(64))
    assert batch["max_active"] == 16
    assert len(observed_batches) == 4
    assert all(len(body["prompt"]) == 16 for body in observed_batches)
    assert batch["generated_tokens"] == 64 * 32
    assert batch["aggregate_generated_tokens_per_second"] > 0
    assert batch["wave_slot_erase_evidence"] == [[287] * 16 for _ in range(3)]


def test_continuous_batch_pins_recurrent_outputs_to_stable_slots(monkeypatch):
    evaluator = load_evaluator()

    def fake_stream_batch(_base_url, _body, batch_size, _timeout):
        return [{
            "token_ids": [slot] * 32,
            "tokens_predicted": 32,
            "tokens_evaluated": 256,
            "ttft_ms": 1.0,
            "end_to_end_ms": 2.0,
            "timings": {"prompt_per_second": 200.0, "predicted_per_second": 10.0},
            "id_slot": slot,
        } for slot in range(batch_size)]

    monkeypatch.setattr(evaluator, "stream_completion_batch", fake_stream_batch)
    monkeypatch.setattr(evaluator, "erase_all_slots", lambda *_args: [287] * 16)
    batch = evaluator.run_continuous_batch(
        "http://127.0.0.1:1",
        evaluator.completion_request(list(range(256))),
        timeout=10,
    )

    assert [item["id_slot"] for item in batch["requests"]] == list(range(16)) * 4
    assert [item["token_ids"] for item in batch["requests"]] == [
        [request_id % 16] * 32 for request_id in range(64)
    ]


FAKE_SERVER = r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, required=True)
args, _ = parser.parse_known_args()
mode = os.environ["FAKE_MODE"]
requests_path = os.environ["FAKE_REQUESTS"]
comparison_emitted = False

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def body(self):
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size))

    def send_json(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        global comparison_emitted
        body = self.body()
        if self.path == "/tokenize":
            content = body["content"]
            if content.startswith("seed "):
                tokens = list(range(300))
            elif content == "roundtrip-256":
                tokens = list(range(256))
            else:
                tokens = [701, 702]
            self.send_json({"tokens": tokens})
            return
        if self.path == "/detokenize":
            assert body["tokens"] == list(range(256))
            self.send_json({"content": "roundtrip-256"})
            return
        if self.path == "/apply-template":
            assert body["messages"] == [{"role": "user", "content": "Reply with exactly OK and no other text."}]
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            self.send_json({"prompt": "templated semantic prompt"})
            return
        if self.path.startswith("/slots/") and self.path.endswith("?action=erase"):
            slot = int(self.path.split("/", 2)[2].split("?", 1)[0])
            self.send_json({"id_slot": slot, "n_erased": 0})
            return
        if self.path != "/completion":
            self.send_error(404)
            return
        with open(requests_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"mode": mode, "body": body}, sort_keys=True) + "\n")
        batched = isinstance(body["prompt"], list) and body["prompt"] and isinstance(body["prompt"][0], list)
        semantic = isinstance(body["prompt"], str)
        hardware_probe = semantic and body["n_predict"] == 2
        if hardware_probe and mode != "cuda" and not comparison_emitted:
            comparison_emitted = True
            comparison = {
                "event": "captured_layer_comparison",
                "backend": "xrt",
                "kernel": "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                "launch_ordinal": 0,
                "output_hash": "0123456789abcdef",
                "reference_checked_components": 64,
                "comparison_status": "pass",
                "comparison": {
                    "status": "pass",
                    "reference_backend": "scalar_iq1s",
                    "checked_elements": 2,
                    "atol": 1.0e-4,
                    "rtol": 1.0e-3,
                    "max_absolute_error": 0.0,
                    "max_relative_error": 0.0,
                    "reference_outputs": [1.0, -2.0],
                    "actual_outputs": [1.0, -2.0],
                },
            }
            print(json.dumps(comparison, sort_keys=True), file=sys.stderr, flush=True)
        if hardware_probe and os.environ.get("FAKE_ROUTE_EVIDENCE"):
            route_records = [
                {
                    "kernel": "flash_attn_f32",
                    "route": "gpu",
                    "backend": "xrt",
                    "strict": True,
                    "xrt_enabled": True,
                    "hardware_matmul_enabled": False,
                },
                {
                    "kernel": "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii",
                    "route": "cxl_tmatmul",
                    "backend": "xrt",
                    "strict": True,
                    "xrt_enabled": True,
                    "hardware_matmul_enabled": True,
                },
            ]
            with open(os.environ["FAKE_ROUTE_EVIDENCE"], "a", encoding="utf-8") as stream:
                for record in route_records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            per_cu = [2, 1, 1, 0] if os.environ.get("FAKE_XRT_INACTIVE_CU") else [1, 1, 1, 1]
            trace_assembly = "ldv v0, PARAM_INPUT\ntmatmul_import v0\ntmatmul_go PARAM_MATRIX\ntmatmul_export v0\nsv v0, PARAM_OUTPUT\nstall\n"
            trace_instructions = [
                ["ldv", "v0", "PARAM_INPUT"],
                ["tmatmul_import", "v0"],
                ["tmatmul_go", "PARAM_MATRIX"],
                ["tmatmul_export", "v0"],
                ["sv", "v0", "PARAM_OUTPUT"],
                ["stall"],
            ]
            semantic = hashlib.sha256(b"hetgpu-tmatmul-semantic-trace-v1\0")
            for instruction in trace_instructions:
                for token in instruction:
                    semantic.update(token.encode())
                    semantic.update(b"\0")
                semantic.update(b"\n")
            encoded = bytes(range(16))
            program_sha = hashlib.sha256(encoded).hexdigest()
            physical = []
            for index in range(4):
                cache_hit = index >= 2
                physical.append({
                    "request_id": index,
                    "cu_index": index,
                    "stall_code": index + 1,
                    "dispatch_to_stall_ns": 1000 + index,
                    "matrix_key_sha256": f"{index + 1:064x}",
                    "matrix_content_sha256": f"{index + 5:064x}",
                    "matrix_address": 0x1000 + index * 0x1000,
                    "matrix_cache_hit": cache_hit,
                    "matrix_bytes_transferred": 0 if cache_hit else 262144,
                    "trace_mode": mode,
                    "model_context_limit": 262144,
                    "trace_semantic_sha256": semantic.hexdigest(),
                    "trace_assembly_sha256": hashlib.sha256(trace_assembly.encode()).hexdigest(),
                    "replay_safe_program_sha256": program_sha,
                    "trace_assembly": trace_assembly,
                    "trace_instructions": trace_instructions,
                    "encoded_program_sha256": program_sha,
                    "encoded_program_hex": encoded.hex(),
                    "program_address": 0x10000 + index * 0x1000,
                    "program_bytes": len(encoded),
                    "program_cache_hit": cache_hit,
                })
            xrt_record = {
                "event": "au250_xrt_iq1s_completed",
                "evidence": {
                    "backend": "xrt",
                    "comparison_status": "pass",
                    "submission_count": 4,
                    "completion_count": 4,
                    "per_cu_submissions": per_cu,
                    "per_cu_completions": per_cu,
                    "request_ids": [0, 1, 2, 3],
                    "stall_codes": [1, 1, 1, 1],
                    "raw_min": -16,
                    "raw_max": 19,
                    "reference_checked_components": 64,
                    "resident_matrix_hits": 2,
                    "resident_matrix_misses": 2,
                    "resident_matrix_bytes_transferred": 2 * 262144,
                    "program_cache_hits": 2,
                    "program_cache_misses": 2,
                    "host_pack_hits": 2,
                    "host_pack_misses": 2,
                    "host_pack_bytes_built": 2 * 262144,
                    "physical_completions": physical,
                },
            }
            with open(os.environ["FAKE_XRT_EVIDENCE"], "a", encoding="utf-8") as stream:
                stream.write(json.dumps(xrt_record, sort_keys=True) + "\n")
        tokens = ([777, 778] if hardware_probe else [777]) if semantic else list(range(1000, 1000 + body["n_predict"]))
        pieces = (["OK", ""] if hardware_probe else ["OK"]) if semantic else [f"t{index}" for index in range(body["n_predict"])]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        result_indices = range(len(body["prompt"])) if batched else range(1)
        for result_index in result_indices:
            for processed in (0, 128, 256):
                progress = {
                    "index": result_index,
                    "id_slot": result_index,
                    "content": "",
                    "tokens": [0],
                    "stop": False,
                    "tokens_predicted": 0,
                    "tokens_evaluated": 2 if semantic else 256,
                    "prompt_progress": {"total": 256, "cache": 0, "processed": processed, "time_ms": processed},
                }
                self.wfile.write(("data: " + json.dumps(progress) + "\n\n").encode())
            for token_index, (token, piece) in enumerate(zip(tokens, pieces)):
                payload = {
                    "index": result_index,
                    "id_slot": result_index,
                    "content": piece,
                    "tokens": [token],
                    "stop": False,
                    "tokens_predicted": token_index + 1,
                    "tokens_evaluated": 2 if semantic else 256,
                }
                self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            final = {
                "index": result_index,
                "id_slot": result_index,
                "content": "",
                "tokens": [],
                "stop": True,
                "tokens_predicted": len(tokens),
                "tokens_evaluated": 2 if semantic else 256,
                "timings": {
                    "prompt_ms": 1280.0,
                    "prompt_per_second": 200.0,
                    "predicted_ms": 3200.0,
                    "predicted_per_second": 10.0,
                },
            }
            self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())

print("llama_model_load_tensors: offloaded 65/65 layers to GPU", file=sys.stderr, flush=True)
print("llama_perf_context_print:        load time =    1234.00 ms", file=sys.stderr, flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_mode(tmp_path, mode, server, requests, profile="full"):
    proof = tmp_path / mode
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    env = os.environ.copy()
    env.update({"FAKE_MODE": mode, "FAKE_REQUESTS": str(requests)})
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--mode", mode,
            "--profile", profile,
            "--server", str(server),
            "--model", str(model),
            "--prompt-seed", str(tmp_path / "seed.txt"),
            "--proof-dir", str(proof),
            "--port", str(free_port()),
            "--threads", "4",
            "--model-size", "5",
            "--model-sha256", hashlib.sha256(model.read_bytes()).hexdigest(),
            "--llama-revision", "925e1179947ea0c0ebfb0032df18af3a729822be",
            "--binary-sha256", hashlib.sha256(server.read_bytes()).hexdigest(),
            "--health-fixture", str(tmp_path / "health.txt"),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((proof / f"{mode}.json").read_text())


def test_evaluator_profiles_are_explicit_and_fixed():
    evaluator = load_evaluator()
    assert evaluator.PROFILES == {
        "one-token": {
            "request_count": 1,
            "max_active": 1,
            "tokens_per_request": 1,
            "measurements": 1,
            "warmups": 0,
        },
        "full": {
            "request_count": 64,
            "max_active": 16,
            "tokens_per_request": 32,
            "measurements": 3,
            "warmups": 1,
        },
    }


def test_fake_server_preserves_identical_requests_and_fixed_counts(tmp_path):
    server = tmp_path / "fake-server.py"
    make_executable(server, FAKE_SERVER)
    (tmp_path / "seed.txt").write_text("seed " * 300, encoding="utf-8")
    (tmp_path / "health.txt").write_text("Level 0 : 0x0 (GOOD)\n", encoding="utf-8")
    requests = tmp_path / "requests.jsonl"

    cuda = run_mode(tmp_path, "cuda", server, requests)
    handwritten = run_mode(tmp_path, "handwritten", server, requests)
    compiler = run_mode(tmp_path, "compiler", server, requests)

    modes = (cuda, handwritten, compiler)
    assert all(len(record["measurements"]) == 3 for record in modes)
    assert all(record["request_count"] == 64 for record in modes)
    assert all(record["max_active_requests"] == 16 for record in modes)
    assert all(record["generated_tokens_per_request"] == 32 for record in modes)
    assert all(record["prompt_token_ids"] == list(range(256)) for record in modes)
    assert all(record["generated_token_ids"] == list(range(1000, 1032)) for record in modes)
    assert all(len(record["generated_token_ids_by_request"]) == 64 for record in modes)
    assert all(record["token_equivalence_measurement"] == 0 for record in modes)
    assert all(record["slot_ids_by_request"] == list(range(16)) * 4 for record in modes)
    assert all(record["slot_erase_evidence"] == [[0] * 16 for _ in range(4)] for record in modes)
    assert all(record["wave_slot_erase_evidence"] == [[[0] * 16] * 3] * 4 for record in modes)
    assert all(record["semantic"] == {"text": "OK", "token_ids": [777]} for record in modes)
    assert all(record["hardware_probe"]["token_ids"] == [777, 778] for record in modes)
    assert cuda["sampled_ffn_comparison"] is None
    assert handwritten["sampled_ffn_comparison"]["checked_elements"] == 2
    assert compiler["sampled_ffn_comparison"]["checked_elements"] == 2
    assert cuda["placement"] == {"all_layers_on_gpu": True, "cpu_layers": 0}
    for mode, record in zip(("cuda", "handwritten", "compiler"), modes):
        command = json.loads((tmp_path / mode / "command.json").read_text(encoding="utf-8"))
        assert "--no-warmup" in command
        assert command[command.index("--parallel") + 1] == "16"
        assert command[command.index("--ctx-size") + 1] == "8192"
        assert command[command.index("--cache-ram") + 1] == "0"
        assert command[command.index("--flash-attn") + 1] == "on"
        assert "--no-cache-prompt" in command
        assert command[command.index("--slot-save-path") + 1].endswith(f"/{mode}/slot-state")
        assert record["context_tokens_per_request"] == 512
        assert record["server_context_tokens"] == 8192

    records = [json.loads(line) for line in requests.read_text().splitlines()]
    assert len(records) == 3 * (2 + 4 * 4)
    cuda_bodies = [item["body"] for item in records if item["mode"] == "cuda"]
    handwritten_bodies = [item["body"] for item in records if item["mode"] == "handwritten"]
    compiler_bodies = [item["body"] for item in records if item["mode"] == "compiler"]
    assert cuda_bodies[:2] == handwritten_bodies[:2] == compiler_bodies[:2]
    canonical = lambda bodies: sorted(json.dumps(body, sort_keys=True) for body in bodies[2:])
    assert canonical(cuda_bodies) == canonical(handwritten_bodies) == canonical(compiler_bodies)
    semantic = cuda_bodies[0]
    assert semantic["prompt"] == "templated semantic prompt"
    assert semantic["n_predict"] == 1
    probe = cuda_bodies[1]
    assert probe["prompt"] == "templated semantic prompt"
    assert probe["n_predict"] == 2
    timed = cuda_bodies[2:]
    assert len(timed) == 4 * 4
    assert all(body["prompt"] == [list(range(256))] * 16 for body in timed)
    assert all(body["n_predict"] == 32 for body in timed)
    assert all(body["temperature"] == 0.0 and body["seed"] == 42 for body in timed)
    assert all(body["ignore_eos"] is True for body in timed)
    assert all(body["cache_prompt"] is False for body in timed)


def test_one_token_profile_records_exact_token_without_e2e_tps_label(tmp_path):
    server = tmp_path / "fake-server.py"
    make_executable(server, FAKE_SERVER)
    (tmp_path / "seed.txt").write_text("seed " * 300, encoding="utf-8")
    (tmp_path / "health.txt").write_text("Level 0 : 0x0 (GOOD)\n", encoding="utf-8")
    requests = tmp_path / "requests.jsonl"

    record = run_mode(tmp_path, "cuda", server, requests, profile="one-token")
    assert record["profile"] == {
        "name": "one-token",
        "request_count": 1,
        "max_active": 1,
        "tokens_per_request": 1,
        "measurements": 1,
        "warmups": 0,
    }
    assert record["generated_token_ids"] == [1000]
    assert record["generated_token_ids_by_request"] == [[1000]]
    assert record["measurements"][0]["generated_tokens"] == 1
    assert "aggregate_generated_tokens_per_second" not in record["measurements"][0]
    assert "e2e_tps" not in json.dumps(record).lower()


def test_rejects_non_roundtripping_prompt(tmp_path):
    evaluator_source = EVALUATOR.read_text(encoding="utf-8")
    assert "retoken" in evaluator_source.lower()
    assert "exactly 256" in evaluator_source.lower()


def test_parse_load_ms_uses_verbose_server_timestamps():
    evaluator = load_evaluator()
    log = "\n".join(
        [
            "0.00.120.675 I srv    load_model: loading model '/models/qwen.gguf'",
            "1.50.407.110 I srv  llama_server: model loaded",
        ]
    )

    assert evaluator.parse_load_ms(log) == pytest.approx(110_286.435)


def metric(median, minimum, maximum, stdev):
    return {
        "median": median,
        "min": minimum,
        "max": maximum,
        "population_stdev": stdev,
        "cv": stdev / median,
    }


def test_render_report_uses_only_validated_metrics_and_cu_counts(tmp_path):
    evaluator = load_evaluator()
    cuda_metrics = {
        "prompt_tokens_per_second": metric(20.0, 19.0, 21.0, 0.5),
        "generation_tokens_per_second": metric(5.0, 4.5, 5.5, 0.2),
        "ttft_ms": metric(100.0, 90.0, 110.0, 4.0),
        "end_to_end_ms": metric(7000.0, 6900.0, 7100.0, 50.0),
        "model_load_ms": metric(1000.0, 1000.0, 1000.0, 0.0),
    }
    hybrid_metrics = {
        "prompt_tokens_per_second": metric(10.0, 9.0, 11.0, 0.4),
        "generation_tokens_per_second": metric(4.0, 3.5, 4.5, 0.1),
        "ttft_ms": metric(200.0, 190.0, 210.0, 5.0),
        "end_to_end_ms": metric(8000.0, 7900.0, 8100.0, 60.0),
        "model_load_ms": metric(1000.0, 1000.0, 1000.0, 0.0),
    }
    normalized = {
        "schema_version": 1,
        "status": "pass",
        "token_ids_match": True,
        "eligible_route_coverage": 1.0,
        "all_cus_active": True,
        "modes": {
            "cuda": {"measurements": 5, "metrics": cuda_metrics},
            "hybrid": {
                "measurements": 5,
                "metrics": hybrid_metrics,
                "routes": {"eligible": 8, "handled": 8, "fallback": 0, "error": 0},
                "xrt": {"per_cu_completions": [4, 3, 2, 1]},
            },
        },
    }
    report = evaluator.render_report(normalized, tmp_path / "proof")
    assert "Active CUs: 4/4" in report
    assert "| Prompt tokens/s | 20 | 10 | 0.5 |" in report
    assert "| Time to first token (ms) | 100 | 200 | 2 |" in report
    assert "Eligible expert operations handled by AU250: 100%" in report
    table_rows = [line for line in report.splitlines() if line.startswith("| ")][2:]
    assert len(table_rows) == 4
    for row in table_rows:
        numeric_cells = [cell.strip() for cell in row.split("|")[2:5]]
        assert all(math.isfinite(float(cell)) for cell in numeric_cells)


def test_render_report_refuses_nonpassing_proof():
    evaluator = load_evaluator()
    with pytest.raises(evaluator.EvaluationError):
        evaluator.render_report({"status": "fail"}, Path("proof"))


def iq1s_report_proof():
    cuda_metrics = {
        "prompt_tokens_per_second": metric(20.0, 19.0, 21.0, 0.5),
        "generation_tokens_per_second": metric(5.0, 4.5, 5.5, 0.2),
        "ttft_ms": metric(100.0, 90.0, 110.0, 4.0),
        "end_to_end_ms": metric(7000.0, 6900.0, 7100.0, 50.0),
        "model_load_ms": metric(1000.0, 1000.0, 1000.0, 0.0),
    }
    hybrid_metrics = {
        "prompt_tokens_per_second": metric(10.0, 9.0, 11.0, 0.4),
        "generation_tokens_per_second": metric(4.0, 3.5, 4.5, 0.1),
        "ttft_ms": metric(200.0, 190.0, 210.0, 5.0),
        "end_to_end_ms": metric(8000.0, 7900.0, 8100.0, 60.0),
        "model_load_ms": metric(1000.0, 1000.0, 1000.0, 0.0),
    }
    return {
        "schema_version": 3,
        "status": "pass",
        "model": {
            "size": 94155830880,
            "sha256": "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568",
            "architecture": "qwen35moe",
            "llama_revision": "925e1179947ea0c0ebfb0032df18af3a729822be",
            "binary_sha256": "a" * 64,
        },
        "model_audit": {
            "routed_expert_count": 180,
            "routed_expert_types": {
                "IQ1_S": 141,
                "IQ2_XXS": 24,
                "IQ3_S": 4,
                "MXFP4": 11,
            },
            "tq1_0_total": 0,
            "non_expert_iq1s": [],
        },
        "token_ids_match": True,
        "eligible_route_coverage": 1.0,
        "tensor_eligibility_coverage": 141 / 180,
        "all_cus_active": True,
        "modes": {
            "cuda": {"measurements": 3, "metrics": cuda_metrics},
            "handwritten": {
                "measurements": 3,
                "metrics": hybrid_metrics,
                "sampled_ffn_comparison": {
                    "status": "pass", "reference_backend": "scalar_iq1s",
                    "checked_elements": 2, "atol": 1e-4, "rtol": 1e-3,
                    "max_absolute_error": 1e-6, "max_relative_error": 1e-6,
                    "reference_outputs": [1.0, 2.0], "actual_outputs": [1.000001, 2.0],
                    "phase": "pre_timed", "kernel": "iq1s",
                },
                "routes": {"eligible": 8, "handled": 8, "fallback": 0, "error": 0},
                "xrt": {
                    "per_cu_completions": [4, 3, 2, 1],
                    "submission_count": 10,
                    "completion_count": 10,
                },
            },
            "compiler": {
                "measurements": 3,
                "metrics": hybrid_metrics,
                "sampled_ffn_comparison": {
                    "status": "pass", "reference_backend": "scalar_iq1s",
                    "checked_elements": 2, "atol": 1e-4, "rtol": 1e-3,
                    "max_absolute_error": 2e-6, "max_relative_error": 2e-6,
                    "reference_outputs": [1.0, 2.0], "actual_outputs": [1.000002, 2.0],
                    "phase": "pre_timed", "kernel": "iq1s",
                },
                "routes": {"eligible": 8, "handled": 8, "fallback": 0, "error": 0},
                "xrt": {
                    "per_cu_completions": [4, 3, 2, 1],
                    "submission_count": 10,
                    "completion_count": 10,
                },
            },
        },
        "sampled_ffn_within_tolerance": True,
    }


def test_iq1s_report_states_mixed_format_and_physical_boundary():
    evaluator = load_evaluator()
    report = evaluator.render_iq1s_report(iq1s_report_proof(), Path("proof"))
    assert "141/180 routed-expert tensors eligible" in report
    assert "IQ2_XXS, IQ3_S, and MXFP4 remained on CUDA" in report
    assert "Eligible IQ1_S operations handled by AU250: 100%" in report
    assert "Active CUs: 4/4" in report
    assert "64-request continuous batches" in report
    assert "Sampled FFN maximum absolute error" in report
    assert "pure TQ1_0" not in report


@pytest.mark.parametrize(
    "mutation",
    (
        lambda proof: proof.update(status="fail"),
        lambda proof: proof.update(eligible_route_coverage=0.5),
        lambda proof: proof.update(all_cus_active=False),
        lambda proof: proof.pop("model_audit"),
        lambda proof: proof["modes"]["handwritten"]["metrics"][
            "generation_tokens_per_second"
        ].update(median=float("nan")),
    ),
)
def test_iq1s_report_rejects_unqualified_normalized_proof(mutation):
    evaluator = load_evaluator()
    normalized = iq1s_report_proof()
    mutation(normalized)
    with pytest.raises(evaluator.EvaluationError):
        evaluator.render_iq1s_report(normalized, Path("proof"))


def iq1s_route(kernel, route="cxl_tmatmul", hardware=True):
    return {
        "kernel": kernel,
        "route": route,
        "backend": "xrt",
        "strict": True,
        "xrt_enabled": True,
        "hardware_matmul_enabled": hardware,
    }


def iq1s_xrt_record(**overrides):
    physical = [
        {"request_id": index, "cu_index": index}
        for index in range(4)
    ]
    evidence = {
        "backend": "xrt",
        "comparison_status": "pass",
        "submission_count": 4,
        "completion_count": 4,
        "per_cu_submissions": [1, 1, 1, 1],
        "per_cu_completions": [1, 1, 1, 1],
        "request_ids": [0, 1, 2, 3],
        "stall_codes": [1, 1, 1, 1],
        "raw_min": -11,
        "raw_max": 17,
        "reference_checked_components": 64,
        "resident_matrix_hits": 2,
        "resident_matrix_misses": 2,
        "resident_matrix_bytes_transferred": 2 * 262144,
        "program_cache_hits": 2,
        "program_cache_misses": 2,
        "host_pack_hits": 2,
        "host_pack_misses": 2,
        "host_pack_bytes_built": 2 * 262144,
        "physical_completions": physical,
    }
    evidence.update(overrides)
    return {"event": "au250_xrt_iq1s_completed", "evidence": evidence}


def valid_iq1s_routes():
    return [
        iq1s_route("_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii"),
        iq1s_route("_Z9mul_mat_qIL9ggml_type16ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii", "gpu", False),
        iq1s_route("flash_attn_f32", "gpu", False),
        iq1s_route("mul_mat_q_stream_k_fixup_ggml_type19", "gpu", False),
    ]


def test_parse_iq1s_routing_selects_only_exact_type19_matmul_and_physical_xrt():
    evaluator = load_evaluator()

    routes, xrt, attention = evaluator.parse_iq1s_routing(
        valid_iq1s_routes(), [iq1s_xrt_record()]
    )

    assert routes == {
        "eligible": 1,
        "handled": 1,
        "fallback": 0,
        "error": 0,
        "eligible_kernels": [
            "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii"
        ],
    }
    assert xrt["submission_count"] == xrt["completion_count"] == 4
    assert xrt["per_cu_submissions"] == xrt["per_cu_completions"] == [1, 1, 1, 1]
    assert xrt["request_ids"] == [(1 << 32) + index for index in range(4)]
    assert attention == 1


@pytest.mark.parametrize(
    "mutation",
    ["fallback", "reject", "missing_xrt", "xrt_without_route", "duplicate_id", "wrong_cu", "zero_stall", "raw_overflow"],
)
def test_parse_iq1s_routing_rejects_incomplete_or_invalid_evidence(mutation):
    evaluator = load_evaluator()
    routes = valid_iq1s_routes()
    xrt = [iq1s_xrt_record()]
    if mutation == "fallback":
        routes[0]["route"] = "gpu"
        routes[0]["hardware_matmul_enabled"] = False
    elif mutation == "reject":
        routes[0]["route"] = "reject"
        routes[0]["hardware_matmul_enabled"] = False
    elif mutation == "missing_xrt":
        xrt = []
    elif mutation == "xrt_without_route":
        routes = routes[1:]
    elif mutation == "duplicate_id":
        xrt = [iq1s_xrt_record(request_ids=[0, 1, 1, 3])]
    elif mutation == "wrong_cu":
        xrt = [iq1s_xrt_record(per_cu_completions=[2, 1, 1, 0])]
    elif mutation == "zero_stall":
        xrt = [iq1s_xrt_record(stall_codes=[1, 1, 0, 1])]
    elif mutation == "raw_overflow":
        xrt = [iq1s_xrt_record(raw_max=4097)]

    with pytest.raises(evaluator.EvaluationError):
        evaluator.parse_iq1s_routing(routes, xrt)


def test_iq1s_jsonl_reader_rejects_malformed_and_empty_files(tmp_path):
    evaluator = load_evaluator()
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"route":\n', encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")

    with pytest.raises(evaluator.EvaluationError, match="invalid IQ1_S route evidence"):
        evaluator.load_jsonl_records(malformed, "IQ1_S route evidence", required=True)
    with pytest.raises(evaluator.EvaluationError, match="is empty"):
        evaluator.load_jsonl_records(empty, "IQ1_S XRT evidence", required=True)


def run_iq1s_mode(tmp_path, inactive_cu=False):
    server = tmp_path / "fake-iq1s-server.py"
    make_executable(server, FAKE_SERVER)
    (tmp_path / "seed.txt").write_text("seed " * 300, encoding="utf-8")
    (tmp_path / "health.txt").write_text("Level 0 : 0x0 (GOOD)\n", encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    audit = tmp_path / "model-tensor-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "model_sha256": model_hash,
                "architecture": "qwen35moe",
                "routed_expert_count": 180,
                "routed_expert_types": {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11},
                "tq1_0_total": 0,
                "non_expert_iq1s": [],
            }
        ),
        encoding="utf-8",
    )
    proof = tmp_path / "handwritten-iq1s"
    routes = proof / "routes.jsonl"
    xrt = proof / "xrt.jsonl"
    requests = tmp_path / "iq1s-requests.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_MODE": "handwritten",
            "FAKE_REQUESTS": str(requests),
            "FAKE_ROUTE_EVIDENCE": str(routes),
            "FAKE_XRT_EVIDENCE": str(xrt),
        }
    )
    if inactive_cu:
        env["FAKE_XRT_INACTIVE_CU"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--mode", "handwritten",
            "--profile", "full",
            "--evidence-kind", "iq1s",
            "--server", str(server),
            "--model", str(model),
            "--prompt-seed", str(tmp_path / "seed.txt"),
            "--proof-dir", str(proof),
            "--port", str(free_port()),
            "--threads", "4",
            "--model-size", str(model.stat().st_size),
            "--model-sha256", model_hash,
            "--binary-sha256", hashlib.sha256(server.read_bytes()).hexdigest(),
            "--health-fixture", str(tmp_path / "health.txt"),
            "--route-evidence", str(routes),
            "--xrt-evidence", str(xrt),
            "--model-audit", str(audit),
            "--require-routing-evidence",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    return result, proof, requests


def test_iq1s_semantic_hardware_gate_passes_before_timed_requests(tmp_path):
    result, proof, requests = run_iq1s_mode(tmp_path)
    assert result.returncode == 0, result.stderr
    record = json.loads((proof / "handwritten.json").read_text(encoding="utf-8"))
    assert record["schema_version"] == 2
    assert record["semantic"]["token_ids"] == [777]
    assert record["hardware_probe"]["token_ids"] == [777, 778]
    assert record["semantic_hardware_gate"]["routes"]["handled"] == 1
    assert record["semantic_hardware_gate"]["xrt"]["per_cu_completions"] == [1, 1, 1, 1]
    assert record["sampled_ffn_comparison"]["status"] == "pass"
    assert record["model_audit_sha256"] == hashlib.sha256(
        (tmp_path / "model-tensor-audit.json").read_bytes()
    ).hexdigest()
    assert len(requests.read_text(encoding="utf-8").splitlines()) == 2 + 4 * 4


def test_iq1s_semantic_hardware_gate_rejects_inactive_cu_before_warmup(tmp_path):
    result, proof, requests = run_iq1s_mode(tmp_path, inactive_cu=True)
    assert result.returncode != 0
    assert "all four CUs" in result.stderr
    assert not (proof / "handwritten.json").exists()
    assert len(requests.read_text(encoding="utf-8").splitlines()) == 2
