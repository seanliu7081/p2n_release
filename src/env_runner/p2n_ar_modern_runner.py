"""New policy boundary adapters; existing simulator runners remain unchanged."""
from src.common.p2n_ar_modern_capabilities import policy_capability
from src.env_runner.byte_observation_policy import ByteObservationPolicy  # noqa: F401
from src.env_runner.executed_action_runner import LiberoExecutedPastRunner
from src.env_runner.state_history_runner import LiberoStateHistoryRunner


class P2NARModernLiberoRunner(LiberoExecutedPastRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement"):
            raise ValueError("This runner requires execution acknowledgement")
        if policy_capability(policy, "requires_state_history"):
            raise ValueError("State-history policy requires P2NARModernStateGateLiberoRunner")
        return super().run(ByteObservationPolicy(policy), **kwargs)


class P2NARModernStateGateLiberoRunner(LiberoStateHistoryRunner):
    def run(self, policy, **kwargs):
        if not policy_capability(policy, "requires_execution_acknowledgement") or not policy_capability(policy, "requires_state_history"):
            raise ValueError("Gate runner requires measured state history and execution acknowledgement")
        return super().run(ByteObservationPolicy(policy), **kwargs)
