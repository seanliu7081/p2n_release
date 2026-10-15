"""Capability-based runner adapters for flow; execution feedback is unchanged."""
from src.env_runner.p2n_ar_modern_runner import P2NARModernLiberoRunner, P2NARModernStateGateLiberoRunner

class P2NActionFlowLiberoRunner(P2NARModernLiberoRunner):
    pass

class P2NStateGateActionFlowLiberoRunner(P2NARModernStateGateLiberoRunner):
    pass
