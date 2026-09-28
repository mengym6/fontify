"""Strict loading for the unconditioned Fontify baseline."""


def load_baseline_checkpoint(model, checkpoint):
    """Reject removed conditioning weights instead of silently ignoring them."""
    config = checkpoint.get("stage3_config") or {}
    state = checkpoint["model"]
    if config.get("style_mode", "off") != "off" or any(
        key.startswith("style_conditioner.") for key in state
    ):
        raise ValueError("FiLM checkpoints are not supported by the baseline")
    model.load_state_dict(state, strict=True)
    return model
