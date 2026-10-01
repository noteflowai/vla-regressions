"""Validate the official X-VLA precision configuration before constructing a model."""
from copy import deepcopy


def configure_precision(config, pipeline):
    if (config.type != "xvla" or pipeline.get("family") != "xvla"
            or pipeline.get("update") not in ("baseline_reload", "bf16")
            or pipeline.get("precision") != {
                "baseline_reload": "float32", "bf16": "bfloat16"}[pipeline["update"]]
            or config.num_denoising_steps != 10 or config.chunk_size != 30
            or config.n_action_steps != 30 or config.n_obs_steps != 1):
        raise ValueError("Precision or cadence differs from the frozen first-study pipeline")
    changed = deepcopy(config)
    changed.dtype = pipeline["precision"]
    if hasattr(changed, "compile_model"):
        changed.compile_model = False
    return changed


def verify_precision(policy, precision):
    histogram = {}
    for parameter in policy.parameters():
        if parameter.is_floating_point():
            key = str(parameter.dtype).removeprefix("torch.")
            histogram[key] = histogram.get(key, 0) + parameter.numel()
    if not histogram or set(histogram) != {precision}:
        raise ValueError("Loaded floating parameter dtypes differ from declared precision")
    return histogram
