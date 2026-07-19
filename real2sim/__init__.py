"""Real2Sim package — lazy top-level re-exports.

`from real2sim import Real2SimPipeline, PipelineConfig, PoseOptimizer` still works,
but the heavy modules (torch / PyTorch3D / Genesis) are only imported when one of
those names is actually accessed. This lets `python -m real2sim.io.paths` run in
a bare env without dragging in the optimizer stack.
"""

def __getattr__(name):
    if name == "Real2SimPipeline":
        from .pipeline import Real2SimPipeline
        return Real2SimPipeline
    if name == "PipelineConfig":
        from .config import PipelineConfig
        return PipelineConfig
    if name == "PoseOptimizer":
        from .pose.optimizer import PoseOptimizer
        return PoseOptimizer
    raise AttributeError(f"module 'real2sim' has no attribute {name!r}")


__all__ = ["Real2SimPipeline", "PipelineConfig", "PoseOptimizer"]
