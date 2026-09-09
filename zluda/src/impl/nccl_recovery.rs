#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaultClass {
    Transient,
    Degraded,
    Permanent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryAction {
    Retry,
    Migrate,
    Replace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NcclRecoveryDecision {
    pub(crate) class: FaultClass,
    pub(crate) action: RecoveryAction,
    pub(crate) reason: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcclRecoveryPhase {
    Healthy,
    Isolated,
    Restoring,
    Reintegrated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NcclRecoveryState {
    phase: NcclRecoveryPhase,
    original_nranks: i32,
    failed_rank: Option<i32>,
    replacement_rank: Option<i32>,
    active_ranks: Vec<i32>,
}

impl NcclRecoveryState {
    pub(crate) fn healthy(nranks: i32) -> Self {
        let active_ranks = if nranks > 0 {
            (0..nranks).collect()
        } else {
            Vec::new()
        };
        Self {
            phase: NcclRecoveryPhase::Healthy,
            original_nranks: nranks.max(0),
            failed_rank: None,
            replacement_rank: None,
            active_ranks,
        }
    }

    pub(crate) fn phase(&self) -> NcclRecoveryPhase {
        self.phase
    }

    pub(crate) fn active_ranks(&self) -> &[i32] {
        &self.active_ranks
    }

    pub(crate) fn on_failure(
        mut self,
        failed_rank: i32,
        class: FaultClass,
    ) -> Result<Self, String> {
        validate_rank(self.original_nranks, failed_rank, "failed")?;
        if class == FaultClass::Transient {
            return Ok(self);
        }
        self.phase = NcclRecoveryPhase::Isolated;
        self.failed_rank = Some(failed_rank);
        self.replacement_rank = None;
        self.active_ranks = (0..self.original_nranks)
            .filter(|rank| *rank != failed_rank)
            .collect();
        Ok(self)
    }

    pub(crate) fn with_replacement(mut self, replacement_rank: i32) -> Result<Self, String> {
        let failed_rank = self
            .failed_rank
            .ok_or_else(|| "replacement requested before permanent failure".to_string())?;
        self.active_ranks =
            plan_replacement_ring(self.original_nranks, failed_rank, replacement_rank)?;
        self.phase = NcclRecoveryPhase::Restoring;
        self.replacement_rank = Some(replacement_rank);
        Ok(self)
    }

    pub(crate) fn restore_complete(mut self) -> Result<Self, String> {
        if self.phase != NcclRecoveryPhase::Restoring {
            return Err("restore completion requires restoring phase".to_string());
        }
        self.phase = NcclRecoveryPhase::Reintegrated;
        Ok(self)
    }
}

pub(crate) fn plan_replacement_ring(
    nranks: i32,
    failed_rank: i32,
    replacement_rank: i32,
) -> Result<Vec<i32>, String> {
    validate_rank(nranks, failed_rank, "failed")?;
    if (0..nranks).contains(&replacement_rank) && replacement_rank != failed_rank {
        return Err(format!(
            "replacement rank {replacement_rank} collides with existing live rank"
        ));
    }

    Ok((0..nranks)
        .map(|rank| {
            if rank == failed_rank {
                replacement_rank
            } else {
                rank
            }
        })
        .collect())
}

pub(crate) fn classify_nccl_failure(
    result: i32,
    async_error: i32,
    consecutive_errors: u32,
    health_degraded: bool,
) -> NcclRecoveryDecision {
    if health_degraded && result == 0 && async_error == 0 {
        return NcclRecoveryDecision {
            class: FaultClass::Degraded,
            action: RecoveryAction::Migrate,
            reason: "health_degraded",
        };
    }

    if consecutive_errors >= 3 {
        return NcclRecoveryDecision {
            class: FaultClass::Permanent,
            action: RecoveryAction::Replace,
            reason: "repeated_nccl_errors",
        };
    }

    NcclRecoveryDecision {
        class: FaultClass::Transient,
        action: RecoveryAction::Retry,
        reason: "retryable_nccl_error",
    }
}

pub(crate) fn format_active_ring(ranks: &[i32]) -> String {
    ranks
        .iter()
        .map(|rank| rank.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_rank(nranks: i32, rank: i32, label: &str) -> Result<(), String> {
    if nranks <= 0 {
        return Err(format!("nranks must be positive, got {nranks}"));
    }
    if !(0..nranks).contains(&rank) {
        return Err(format!("{label} rank {rank} outside 0..{nranks}"));
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_nccl_classify_failure(
    result: i32,
    async_error: i32,
    consecutive_errors: u32,
    health_degraded: i32,
) -> i32 {
    let decision = classify_nccl_failure(
        result,
        async_error,
        consecutive_errors,
        health_degraded != 0,
    );
    match decision.action {
        RecoveryAction::Retry => 0,
        RecoveryAction::Migrate => 1,
        RecoveryAction::Replace => 2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_nccl_plan_replacement(
    nranks: i32,
    failed_rank: i32,
    replacement_rank: i32,
    out_ranks: *mut i32,
    out_len: i32,
) -> i32 {
    if out_ranks.is_null() || out_len < 0 {
        return -1;
    }
    let Ok(plan) = plan_replacement_ring(nranks, failed_rank, replacement_rank) else {
        return -2;
    };
    if out_len < plan.len() as i32 {
        return -3;
    }
    for (index, rank) in plan.iter().enumerate() {
        *out_ranks.add(index) = *rank;
    }
    plan.len() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_plan_replaces_failed_rank_in_ring_order() {
        let plan = plan_replacement_ring(4, 2, 7).unwrap();

        assert_eq!(plan, vec![0, 1, 7, 3]);
    }

    #[test]
    fn replacement_plan_rejects_invalid_rank_inputs() {
        assert!(plan_replacement_ring(0, 0, 1).is_err());
        assert!(plan_replacement_ring(4, 4, 7).is_err());
        assert!(plan_replacement_ring(4, 1, 3).is_err());
    }

    #[test]
    fn state_machine_isolates_restores_and_reintegrates_replacement() {
        let state = NcclRecoveryState::healthy(4);
        let isolated = state
            .on_failure(2, FaultClass::Permanent)
            .expect("permanent failure should isolate rank");
        assert_eq!(isolated.phase(), NcclRecoveryPhase::Isolated);
        assert_eq!(isolated.active_ranks(), &[0, 1, 3]);

        let restoring = isolated
            .with_replacement(7)
            .expect("replacement should be accepted");
        assert_eq!(restoring.phase(), NcclRecoveryPhase::Restoring);
        assert_eq!(restoring.active_ranks(), &[0, 1, 7, 3]);

        let reintegrated = restoring
            .restore_complete()
            .expect("restore completion should reintegrate replacement");
        assert_eq!(reintegrated.phase(), NcclRecoveryPhase::Reintegrated);
        assert_eq!(reintegrated.active_ranks(), &[0, 1, 7, 3]);
    }

    #[test]
    fn transient_failure_keeps_existing_ring_for_retry() {
        let state = NcclRecoveryState::healthy(3)
            .on_failure(1, FaultClass::Transient)
            .unwrap();

        assert_eq!(state.phase(), NcclRecoveryPhase::Healthy);
        assert_eq!(state.active_ranks(), &[0, 1, 2]);
    }

    #[test]
    fn degraded_health_classifies_as_migration_before_hard_failure() {
        let decision = classify_nccl_failure(0, 0, 0, true);

        assert_eq!(decision.class, FaultClass::Degraded);
        assert_eq!(decision.action, RecoveryAction::Migrate);
        assert_eq!(decision.reason, "health_degraded");
    }

    #[test]
    fn repeated_collective_errors_classify_as_permanent_replacement() {
        let decision = classify_nccl_failure(2, 2, 3, false);

        assert_eq!(decision.class, FaultClass::Permanent);
        assert_eq!(decision.action, RecoveryAction::Replace);
        assert_eq!(decision.reason, "repeated_nccl_errors");
    }

    #[test]
    fn formats_active_ring_for_c_shim_evidence() {
        let plan = plan_replacement_ring(4, 2, 7).unwrap();

        assert_eq!(format_active_ring(&plan), "0,1,7,3");
    }

    #[test]
    fn ffi_replacement_planner_writes_rank_buffer() {
        let mut out = [-1; 4];
        let written =
            unsafe { hetgpu_concordia_nccl_plan_replacement(4, 1, 9, out.as_mut_ptr(), 4) };

        assert_eq!(written, 4);
        assert_eq!(out, [0, 9, 2, 3]);
    }

    #[test]
    fn ffi_failure_classifier_returns_action_code() {
        let action = unsafe { hetgpu_concordia_nccl_classify_failure(2, 2, 3, 0) };

        assert_eq!(action, 2);
    }
}
