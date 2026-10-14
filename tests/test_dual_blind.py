"""Blind panel pixels and pre-training seal checks without loading models."""

from argparse import Namespace

import pytest

from tools.dual_view_protocol import SCHEMA
from tools.export_dual_blind import (
    compose_blind,
    parse_models,
    tensor_image,
    validate_checkpoint_seal,
    validate_manifest,
    validate_sample,
    write_rating_template,
)


def manifest():
    return {
        "schema": SCHEMA,
        "purpose": "fixed_blind_manifest",
        "fixed_pair_sha256": "a" * 64,
        "comparisons": [
            {
                "blind_id": "blind_0000",
                "index": 4,
                "target_path": "glyph.png",
                "ref_path": "ref.png",
                "left_model": "ctrl",
                "right_model": "run_b",
            }
        ],
    }


def test_blind_manifest_seal_and_sample_identity():
    value = manifest()
    validate_manifest(
        value,
        {"fixed_pair_sha256": "a" * 64},
        {"run_b": "run.pth", "ctrl": "ctrl.pth"},
    )
    with pytest.raises(ValueError, match="another fixed"):
        validate_manifest(
            value,
            {"fixed_pair_sha256": "b" * 64},
            {"run_b": "run.pth", "ctrl": "ctrl.pth"},
        )
    with pytest.raises(ValueError, match="seal"):
        validate_checkpoint_seal({"args": Namespace(dual_view=True)}, "a" * 64)
    assert validate_checkpoint_seal(
        {
            "args": Namespace(
                dual_view=True, dual_blind_manifest_sha256="a" * 64
            )
        },
        "a" * 64,
    )
    assert not validate_checkpoint_seal({"args": Namespace()}, "a" * 64)
    sample = {
        "query_target_path": "glyph.png",
        "reference_target_path": "ref.png",
    }
    validate_sample(value["comparisons"][0], sample)
    with pytest.raises(ValueError, match="target"):
        validate_sample(
            value["comparisons"][0], {**sample, "query_target_path": "other"}
        )


def test_panel_follows_sealed_left_right_pixels_and_neutral_csv(tmp_path):
    image = pytest.importorskip("PIL.Image")
    panels = [
        image.new("RGB", (32, 32), color)
        for color in ("white", "black", "red", "blue")
    ]
    canvas = compose_blind(panels, "blind_0000", cell=32)
    assert canvas.size == (140, 68)
    assert canvas.getpixel((73, 50)) == (255, 0, 0)
    assert canvas.getpixel((110, 50)) == (0, 0, 255)
    path = tmp_path / "blind_0000.png"
    canvas.save(path)
    assert image.open(path).size == (140, 68)
    csv = tmp_path / "ratings.csv"
    write_rating_template(csv, manifest()["comparisons"])
    contents = csv.read_text()
    assert "blind_id,structure_preference,pen_preference" in contents
    assert "blind_0000" in contents
    assert "run_b" not in contents and "ctrl" not in contents
    with pytest.raises(FileExistsError):
        write_rating_template(csv, manifest()["comparisons"])


def test_models_must_use_unambiguous_unique_labels():
    assert parse_models(["run_b=x.pth", "ctrl=y.pth"]) == {
        "run_b": "x.pth",
        "ctrl": "y.pth",
    }
    for entries in (["x.pth"], ["run_b=x.pth", "run_b=y.pth"]):
        with pytest.raises(ValueError):
            parse_models(entries)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_predictions_are_rejected_before_image_cast(nonfinite):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="nonfinite"):
        tensor_image(torch.full((3, 16, 16), nonfinite))
