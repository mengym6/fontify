"""Strict checkpoint reconstruction for legacy and dual-view Fontify models."""

from collections.abc import Mapping


def checkpoint_args(checkpoint):
    """Accept both argparse namespaces and JSON-compatible argument dictionaries."""
    args = checkpoint.get("args")
    if args is None:
        return {}
    return dict(args) if isinstance(args, Mapping) else vars(args)


def checkpoint_is_dual(checkpoint):
    """Require architecture metadata and dual module keys to agree."""
    args = checkpoint_args(checkpoint)
    config = checkpoint.get("dual_view_config")
    configured = bool(args.get("dual_view", False) or config)
    keys = checkpoint["model"]
    has_dual = any(key.startswith(("dual_head.", "dual_fusion.")) for key in keys)
    if configured != has_dual:
        raise ValueError("dual_view metadata and model keys disagree")
    return configured


def restore_legacy_jieti(model, checkpoint):
    """Recreate saved loss buffers; never delete them to make strict loading pass."""
    if not any(key.startswith("jieti_loss_mod.") for key in checkpoint["model"]):
        return
    args = checkpoint_args(checkpoint)
    if not hasattr(model, "enable_jieti"):
        raise ValueError("Model cannot reconstruct saved jieti_loss_mod buffers")
    model.enable_jieti(
        args.get("jieti_alpha_jt", 1.0),
        args.get("jieti_w", 1.0),
        k_max=args.get("jieti_k_max", 4),
        pool=args.get("jieti_pool", 224),
        soft_fg=args.get("jieti_soft_fg", "linear"),
        sigmoid_scale=args.get("jieti_sigmoid_scale", 10.0),
        pred_mass_ratio=args.get("jieti_pred_mass_ratio", 0.1),
        w_centroid=args.get("jieti_w_centroid", 1.0),
        w_logsigma=args.get("jieti_w_logsigma", 1.0),
        w_shape=args.get("jieti_w_shape", 1.0),
    )


def load_fontify_checkpoint(model, checkpoint, allow_dual_initialization=False):
    """Load exactly, allowing only an explicit legacy-to-dual initialization.

    Formal evaluation always keeps ``allow_dual_initialization=False``. Shape
    mismatches, unexpected keys and absent backbone keys are never tolerated.
    """
    config = checkpoint.get("stage3_config") or {}
    state = checkpoint["model"]
    if config.get("style_mode", "off") != "off" or any(
        key.startswith("style_conditioner.") for key in state
    ):
        raise ValueError("FiLM checkpoints are not supported by the baseline")
    restore_legacy_jieti(model, checkpoint)
    if not allow_dual_initialization:
        model.load_state_dict(state, strict=True)
        return model
    if checkpoint_is_dual(checkpoint) or not getattr(model, "dual_view", False):
        raise ValueError("Dual initialization requires legacy weights and dual model")
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    allowed = ("dual_head.", "dual_fusion.", "dual_jieti_loss.")
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise ValueError(
            f"Illegal migration keys: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    # Fill only explicitly allowed new modules and then use a strict load, so
    # incompatible backbone tensor shapes still raise the standard exception.
    merged = dict(expected)
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    return model


def load_baseline_checkpoint(model, checkpoint):
    """Reject removed conditioning weights instead of silently ignoring them."""
    return load_fontify_checkpoint(model, checkpoint)


def build_checkpoint_model(checkpoint, architecture=None, model_module=None):
    """Reconstruct architecture and saved loss buffers before a strict load."""
    args = checkpoint_args(checkpoint)
    dual = checkpoint_is_dual(checkpoint)
    if model_module is None:
        import models_train as model_module
    architecture = architecture or args.get(
        "model", "vit_base_patch16_input896x448_win_dec64_8glb_sl1"
    )
    model = getattr(model_module, architecture)(dual_view=dual)
    model.fusion_lambda = args.get("fusion_lambda", 0.5)
    if dual:
        config = checkpoint.get("dual_view_config") or {}
        if model.fusion_lambda != 0.5:
            raise ValueError("run B requires fusion_lambda=0.5")
        required = {
            "lambda_full": ("dual_lambda_full",),
            "w_j": ("dual_jieti_weight", "jieti_w"),
            "w_h": ("dual_head_weight",),
        }
        for name, alternatives in required.items():
            if name not in config and not any(key in args for key in alternatives):
                raise ValueError(f"Missing dual-view loss metadata: {name}")
        coefficients = dict(config.get("coef") or {})
        for name in ("centroid", "logsigma", "shape"):
            if name not in coefficients:
                argument = f"jieti_w_{name}"
                if argument not in args:
                    raise ValueError(f"Missing dual-view loss coefficient: {name}")
                coefficients[name] = args[argument]
        model.enable_dual_losses(
            lambda_full=config.get("lambda_full", args.get("dual_lambda_full", 0.5)),
            jieti_weight=config.get(
                "w_j", args.get("dual_jieti_weight", args.get("jieti_w", 1.0))
            ),
            head_weight=config.get("w_h", args.get("dual_head_weight", 1.0)),
            w_centroid=coefficients["centroid"],
            w_logsigma=coefficients["logsigma"],
            w_shape=coefficients["shape"],
            pool=config.get("pool", args.get("jieti_pool", 224)),
            soft_fg=config.get("soft_fg", args.get("jieti_soft_fg", "linear")),
            pred_mass_ratio=config.get(
                "pred_mass_ratio", args.get("jieti_pred_mass_ratio", 0.1)
            ),
        )
    if args.get('tip_weight') is not None:
        model.enable_tip(
            args['tip_weight'], args.get('edge_warmup_epochs', 10),
            args.get('loss_warmup_duration', 8),
        )
        model.edge_weight_final = 0.0
        model.style_weight = args.get('style_weight', 14.73)
    control = checkpoint.get("g_control_config")
    has_control = any(key.startswith("g_control.") for key in checkpoint["model"])
    if bool(control) != has_control:
        raise ValueError("G control metadata and model keys disagree")
    if control:
        from util.g_control import enable_g_control

        enable_g_control(model, control)
    scale = checkpoint.get("g_scale_config")
    has_scale = any(key.startswith("g_scale.") for key in checkpoint["model"])
    if bool(scale) != has_scale:
        raise ValueError("G scale metadata and model keys disagree")
    if scale:
        from util.g_scale import enable_g_scale

        enable_g_scale(model, scale)
    return load_fontify_checkpoint(model, checkpoint)
