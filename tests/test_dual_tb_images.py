"""Regressions for Run B TensorBoard image selection and sampling controls."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_dual_model import (
    fontify_class,
    instance_targets,
    sample_inputs,
    small_model,
)
from test_dual_training import PredictionModel, fixed_batch

from util.dual_training import evaluate_dual, training_preview, validation_frame


class RecordingWriter:
    log_dir = "test"

    def __init__(self):
        self.images = []

    def add_image(self, tag, value, step, **kwargs):
        self.images.append((tag, np.array(value), step))

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass


def validation_args(tmp_path, **overrides):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"coef": dict.fromkeys(
        ("centroid", "logsigma", "shape"), 1.0,
    )}), encoding="utf-8")
    settings = dict(
        dual_eval_calibration_json=str(path), jieti_pool=8, jieti_k_max=4,
        jieti_soft_fg="linear", jieti_pred_mass_ratio=0.1,
        vis_every_epoch=True, output_dir=str(tmp_path),
        val_tb_image_limit=0, val_tb_images_per_batch=2, val_tb_image_freq=1,
    )
    settings.update(overrides)
    return SimpleNamespace(**settings)


@pytest.mark.parametrize("limit,per_batch,frequency,epoch,expected", [
    (0, 2, 1, 0, 6), (5, 2, 1, 0, 5), (1, 2, 1, 0, 1),
    (0, 1, 1, 0, 3), (0, 0, 1, 0, 3),
    (0, 2, 2, 0, 0), (0, 2, 2, 1, 6), (0, 2, 0, 0, 6),
])
def test_validation_slider_honors_controls_and_preserves_disk_grid(
    tmp_path, limit, per_batch, frequency, epoch, expected,
):
    args = validation_args(
        tmp_path, val_tb_image_limit=limit,
        val_tb_images_per_batch=per_batch, val_tb_image_freq=frequency,
    )
    batch = fixed_batch()
    loader = [{key: value[[0, 2]] for key, value in batch.items()}
              for _ in range(3)]
    for index, item in enumerate(loader):
        item["image"] += index * 0.1
    writer = RecordingWriter()
    baseline = evaluate_dual(loader, PredictionModel(), "cpu", epoch, args)
    result = evaluate_dual(loader, PredictionModel(), "cpu", epoch, args, writer)
    assert result == baseline
    assert result["n_jt"] == result["n_bf"] == 3
    assert len(writer.images) == expected
    assert [step for _, _, step in writer.images] == list(range(expected))
    assert all(tag == f"epoch:{epoch} val x; im_masked; y; tgt"
               for tag, _, _ in writer.images)
    assert len({frame.tobytes() for _, frame, _ in writer.images}) == expected
    frames = []
    model = PredictionModel()
    for item in loader[:2]:
        masks = model.unpatchify(item["mask"].flatten(1)[..., None].repeat(
            1, 1, model.patch_size ** 2 * 3,
        ))
        frames.extend(validation_frame(
            item["image"][i], item["target"][i], item["image"][i], masks[i],
        ) for i in range(2))
    with Image.open(tmp_path / "vis" / f"epoch_{epoch:03d}.png") as saved:
        np.testing.assert_array_equal(np.array(saved), np.concatenate(frames))


def test_real_tensorboard_events_contain_76_distinct_samples(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    from torch.utils.tensorboard import SummaryWriter

    args = validation_args(tmp_path, val_tb_image_limit=76, vis_every_epoch=False)
    batch = fixed_batch()
    loader = [{key: value[[0, 2]] for key, value in batch.items()}
              for _ in range(39)]
    for index, item in enumerate(loader):
        item["image"][0].fill_(index / 100)
        item["image"][1].fill_((index + 39) / 100)
    log_dir = tmp_path / "logs"
    with SummaryWriter(str(log_dir)) as writer:
        result = evaluate_dual(
            loader, PredictionModel(), "cpu", 0, args, writer,
        )
    assert result["n_jt"] == result["n_bf"] == 39
    events = EventAccumulator(str(log_dir), size_guidance={"images": 0})
    events.Reload()
    tag = "epoch:0 val x; im_masked; y; tgt"
    assert events.Tags()["images"] == [tag]
    images = events.Images(tag)
    assert len(images) == 76
    assert [item.step for item in images] == list(range(76))
    assert len({item.encoded_image_string for item in images}) == 76
    assert all((item.width, item.height) == (32, 16) for item in images)
    assert not (tmp_path / "vis").exists()


def test_nonmain_rank_does_not_write_images(tmp_path, monkeypatch):
    from util import misc

    args = validation_args(tmp_path)
    writer = RecordingWriter()
    monkeypatch.setattr(misc, "is_main_process", lambda: False)
    evaluate_dual([fixed_batch()], PredictionModel(), "cpu", 0, args, writer)
    assert writer.images == []
    assert not (tmp_path / "vis").exists()


def test_training_preview_keeps_legacy_unpatchify_contract():
    mask = torch.zeros(1, 32)
    patches = torch.rand(1, 32, 12)
    model = PredictionModel()
    image, returned_mask = training_preview(model, patches, mask)
    torch.testing.assert_close(image, model.unpatchify(patches))
    assert returned_mask is mask


def test_preview_fields_preserve_original_outputs_and_gradients():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", "458c528:models_train.py"], cwd=root, text=True,
    )
    original = fontify_class(source)(
        img_size=(16, 8), patch_size=2, embed_dim=8, depth=12,
        num_heads=1, use_abs_pos=False, decoder_embed_dim=3, dual_view=True,
    )
    original.enable_dual_losses(pool=8)
    original.improved_edge_detection = lambda value: value
    original._dbg_step = 1
    current = small_model(dual=True)
    original.load_state_dict(current.state_dict(), strict=True)
    inputs = sample_inputs()
    labels = instance_targets([True, False])
    old = original(*inputs, no_gan=True, dual_targets=labels)
    new = current(*inputs, no_gan=True, dual_targets=labels)
    for expected, actual in zip(old[:8], new[:8]):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert not new[-1]["pen_prediction"].requires_grad
    torch.testing.assert_close(new[-1]["pen_mask"], inputs[2].flatten(1))
    old[0].backward()
    new[0].backward()
    for name, parameter in current.named_parameters():
        expected = dict(original.named_parameters())[name].grad
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


def test_actual_engine_previews_pen_forward_without_extra_forward(
    tmp_path, monkeypatch,
):
    import engine_train

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(engine_train, "_write_dual_diagnostics", lambda *x: None)
    model = small_model(dual=True).bfloat16()
    images, targets, mask, valid = sample_inputs()
    batch = dict(image=images, target=targets, mask=mask, valid=valid,
                 **instance_targets([True, False]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    class CpuScaler:
        def __call__(self, loss, optimizer, **kwargs):
            loss.backward()
            optimizer.step()
            return 1.0

        def state_dict(self):
            return {"scale": 1.0}

    forwards, decodes = [], []
    model.register_forward_hook(
        lambda module, args, output: forwards.append(output),
    )
    model.decoder_pred.register_forward_hook(
        lambda module, args, output: decodes.append(output),
    )
    writer = RecordingWriter()
    args = SimpleNamespace(
        dual_view=True, no_gan=True, accum_iter=1, clip_grad=3,
        output_dir=str(tmp_path), lr=0.001, min_lr=0,
        epochs=51, warmup_epochs=10,
    )
    engine_train.train_one_epoch(
        model, [batch], optimizer, torch.device("cpu"), 20,
        CpuScaler(), log_writer=writer, args=args,
    )
    assert len(forwards) == 1 and len(decodes) == 2
    extra = forwards[0][-1]
    pixel_mask = model.unpatchify(
        mask.flatten(1)[0:1, :, None].repeat(1, 1, model.patch_size ** 2 * 3),
    )[0].float()
    expected = validation_frame(
        images[0].bfloat16().float(), targets[0].bfloat16().float(),
        extra["pen_prediction"][0].float(), pixel_mask,
    )
    assert len(writer.images) == 1
    assert writer.images[0][0] == "x; im_masked; y; tgt"
    np.testing.assert_array_equal(writer.images[0][1], expected)
