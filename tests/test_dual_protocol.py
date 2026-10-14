"""Run B protocol math/hash/bootstrap tests and tiny real autograd probes."""

import copy
import json
import math
import random
from argparse import Namespace

import pytest

from tools.dual_report import (
    blind_ratings,
    bootstrap,
    clip_statistics,
    fixed_blind_manifest,
    paired_comparison,
)
from tools.dual_view_protocol import (
    SCHEMA,
    choose_fixed_rows,
    derive_weights,
    gram_norm,
    hash_files,
    norm_stats,
    validate_calibration,
    weight_summary,
    write_new_json,
)


def calibration():
    return {
        "schema": SCHEMA,
        "purpose": "training",
        "status": "validated",
        "lambda_full": 0.5,
        "w_j": 0.2,
        "w_h": 0.1,
        "coef": {"centroid": 1, "logsigma": 1, "shape": 1},
        "edge_weight": 0,
        "norm_rule": "ratio_of_mean_batch_norms",
        **{
            key: "a" * 64
            for key in (
                "train_manifest_sha256",
                "target_cache_sha256",
                "sample_manifest_sha256",
                "warm_checkpoint_sha256",
                "parent_checkpoint_sha256",
                "fixed_pair_sha256",
                "val_manifest_sha256",
                "eval_coef_sha256",
                "s_baseline_sha256",
            )
        },
    }


def test_calibration_rejects_old_or_stale():
    value = calibration()
    assert validate_calibration(value) is value
    for key, bad in (
        ("schema", "legacy"),
        ("purpose", "evaluation"),
        ("status", "diagnostic_required"),
        ("edge_weight", 0.2),
        ("w_j", 0),
        ("w_h", float("nan")),
    ):
        invalid = {**value, key: bad}
        with pytest.raises(ValueError):
            validate_calibration(invalid)
    with pytest.raises(ValueError, match="stale"):
        validate_calibration(value, {"target_cache_sha256": "b" * 64})


def test_hash_stable_across_deployment_and_write_once(tmp_path):
    first, second = tmp_path / "local", tmp_path / "remote"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "a.json").write_text('[{"v": 1}]')
        (directory / "b.json").write_text('[{"v": 2}]')
    assert hash_files(list(first.glob("*.json"))) == hash_files(
        list(second.glob("*.json"))
    )
    (second / "b.json").write_text('[{"v": 3}]')
    assert hash_files(list(first.glob("*.json"))) != hash_files(
        list(second.glob("*.json"))
    )
    output = tmp_path / "result.json"
    write_new_json(output, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new_json(output, {"a": 2})
    assert json.loads(output.read_text()) == {"a": 1}


def test_composed_gradient_norm_includes_cancellation():
    assert gram_norm([[4, -2], [-2, 1]], [1, 2]) == 0
    # Mean sqrt per batch differs from sqrt mean Gram.
    stats = norm_stats([[[1]], [[9]]], [1])
    assert stats["mean"] == 2
    assert stats["mean"] != math.sqrt(5)


def records():
    matrix = [[0.0] * 6 for _ in range(6)]
    for index, norm in enumerate((2, 1, 1, 2, 3, 4)):
        matrix[index][index] = norm**2
    return [
        {
            "group": group,
            "grams": {
                module: copy.deepcopy(matrix)
                for module in ("vit", "decoder", "fusion", "head")
            },
        }
        for group in ("jt", "bf_only", "mixed")
    ]


def test_fit_lambda_then_new_j_and_outer_weights():
    measurements = records()
    fitted = derive_weights(measurements)
    assert fitted["lambda_full"] == 1
    assert fitted["lambda_initial_ratio"] == 0.25
    assert fitted["coef"] == {"centroid": 3, "logsigma": 1.5, "shape": 1}
    summary = weight_summary(measurements, fitted)["jt"]["vit"]
    assert summary["j_base_ratio"] == pytest.approx(0.5)
    assert summary["h_j_ratio"] == pytest.approx(0.5)
    for row in measurements:
        row["grams"]["decoder"][1][1] = 100
    assert derive_weights(measurements)["lambda_full"] == 0.5


def test_zero_component_gradient_requires_diagnosis():
    measurements = records()
    measurements[0]["grams"]["vit"][2][2] = 0
    with pytest.raises(ValueError, match="centroid"):
        derive_weights(measurements)


def test_summary_uses_actual_composed_j_gradient_without_second_inner_weight():
    measurements = records()
    direct = [[0.0] * 4 for _ in range(4)]
    for index, squared in enumerate((4, 1, 81, 16)):
        direct[index][index] = squared
    for row in measurements:
        row["objective_grams"] = {
            module: copy.deepcopy(direct)
            for module in ("vit", "decoder", "fusion", "head")
        }
    weights = {
        "lambda_full": 0.5,
        "w_j": 0.2,
        "w_h": 0.1,
        "coef": dict.fromkeys(("centroid", "logsigma", "shape"), 30),
    }
    summary = weight_summary(measurements, weights)["jt"]["vit"]["norms"]
    assert summary["weighted_j"]["mean"] == pytest.approx(1.8)
    assert summary["weighted_h"]["mean"] == pytest.approx(0.4)


def test_fusion_probe_rejects_zero_projection_and_observes_first_update():
    torch = pytest.importorskip("torch")
    from tools.dual_gradient_probe import check_fusion_open

    model = Namespace(
        dual_fusion=torch.nn.Sequential(
            torch.nn.LayerNorm(2), torch.nn.Linear(2, 2)
        )
    )
    with torch.no_grad():
        model.dual_fusion[-1].weight.zero_()
        model.dual_fusion[-1].bias.zero_()
    assert not check_fusion_open(model)
    with torch.no_grad():
        model.dual_fusion[-1].weight[0, 0] = 1e-5
    assert check_fusion_open(model)


def test_fixed_targets_reference_same_author_same_pool():
    rows = [
        {
            "calligrapher": writer,
            "has_jt": True,
            "has_bf": bf,
            "target_path": f"{writer}_{bf}_{number}.png",
        }
        for writer in ("a", "b")
        for bf in (True, False)
        for number in range(4)
    ]
    samples = choose_fixed_rows(
        rows, [1] * len(rows), range(len(rows)), 12, random.Random(4)
    )
    assert len({row["target_index"] for row in samples}) == 12
    for sample in samples:
        target = rows[sample["target_index"]]
        reference = rows[sample["reference_index"]]
        assert target["calligrapher"] == reference["calligrapher"]
        assert target["has_bf"] == reference["has_bf"]
        assert isinstance(sample["augmentation_seed"], int)


def evaluation(offset=0):
    return {
        "schema": SCHEMA,
        **{
            key: "a" * 64
            for key in (
                "fixed_pair_sha256",
                "val_manifest_sha256",
                "target_cache_sha256",
                "eval_coef_sha256",
            )
        },
        "per_item": [
            {
                "index": i,
                "target_path": f"{i}.png",
                "ref_path": "ref.png",
                "kind": kind,
                "L1": 1 + offset,
                "J": 2 + offset,
                "centroid": 3 + offset,
                "logsigma": 4 + offset,
                "shape": 5 + offset,
            }
            for i, kind in enumerate(("JT", "JT", "BF", "BF"))
        ],
    }


def test_paired_bootstrap_and_cohort_s_not_bf_diluted():
    baseline = evaluation()
    denominator = {**baseline, "L1_JT": 1, "L1_BF": 1, "J": 2}
    result = paired_comparison(evaluation(-0.2), baseline, denominator, 100, 1)
    assert result["groups"]["all"]["L1"]["mean_difference"] == pytest.approx(
        -0.2
    )
    assert result["S"]["mean_difference"] == pytest.approx(-0.5)
    assert bootstrap([2, 2, 2], 100)["ci95"] == [2, 2]
    invalid = evaluation()
    invalid["eval_coef_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="inputs differ"):
        paired_comparison(invalid, baseline, denominator, 100)


def test_clip_statistics_only_update_rows_and_blind_ratings():
    stats = clip_statistics(
        [
            {"pre_clip_grad_norm": 2, "grad_clip_ratio": 1},
            {"pre_clip_grad_norm": 6, "grad_clip_ratio": 0.5},
        ]
    )
    assert stats["trigger_fraction"] == 0.5
    assert stats["clip_ratio"]["median"] == 0.75
    manifest = fixed_blind_manifest(
        evaluation()["per_item"], ["run", "ctrl"], 4, 3
    )
    repeat = fixed_blind_manifest(
        evaluation()["per_item"], ["run", "ctrl"], 4, 3
    )
    assert manifest == repeat
    identifier = manifest["comparisons"][0]["blind_id"]
    result = blind_ratings(
        manifest,
        [
            {
                "blind_id": identifier,
                "structure_preference": "tie",
                "pen_preference": "left",
            }
        ],
    )
    assert result["preferences"]["structure_preference"]["tie"] == 1
    assert len(result["missing"]) == 2


def test_real_autograd_four_modules_same_forward():
    torch = pytest.importorskip("torch")
    from tools.dual_gradient_probe import measure_graph

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Linear(2, 2, bias=False)
            self.decoder_pred = torch.nn.Linear(2, 1, bias=False)
            self.dual_fusion = torch.nn.Linear(2, 2, bias=False)
            self.dual_head = torch.nn.Linear(2, 2, bias=False)

    torch.manual_seed(3)
    model = Tiny()
    token = model.patch_embed(torch.ones(1, 2))
    head = model.dual_head(token)
    pen = model.decoder_pred(token).square().mean()
    full = model.decoder_pred(token + model.dual_fusion(head)).square().mean()
    components = [full * number for number in (1, 2, 3)]
    extra = dict(
        zip(
            ("L_pen", "L_full", "centroid", "logsigma", "shape", "L_H"),
            [pen, full, *components, head.square().mean()],
        )
    )
    extra["J"] = sum(components)
    grams, actual = measure_graph(model, extra)
    assert grams["fusion"][0][0] == 0
    assert grams["head"][0][0] == 0
    assert grams["decoder"][5][5] == 0
    assert grams["fusion"][5][5] == 0
    assert actual["vit"][2][2] == pytest.approx(
        gram_norm(grams["vit"], [0, 0, 1, 1, 1, 0]) ** 2, rel=1e-6
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def training_fixture(tmp_path):
    from tools.dual_view_protocol import cache_hash, sha256_file

    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    train = save("train.json", [])
    val = save("val.json", [])
    fixed = save("fixed.json", [])
    save("cache/index.json", {})
    save("cache/metadata.json", {})
    parent = save("parent.pth", {})
    warm = save("warm.pth", {})
    hashes = {
        "train_manifest_sha256": hash_files([train]),
        "target_cache_sha256": cache_hash(tmp_path / "cache"),
        "parent_checkpoint_sha256": sha256_file(parent),
        "warm_checkpoint_sha256": sha256_file(warm),
        "fixed_pair_sha256": sha256_file(fixed),
        "val_manifest_sha256": hash_files([val]),
    }
    sample = save(
        "samples.json",
        {
            "schema": SCHEMA,
            **hashes,
            "groups": {"jt": [{}] * 64, "bf_only": [{}], "mixed": [{}]},
        },
    )
    hashes["sample_manifest_sha256"] = sha256_file(sample)
    eval_coef = save(
        "eval.json",
        {
            "schema": SCHEMA,
            "purpose": "evaluation",
            "source_epoch": 46,
            "status": "validated",
            "source_role": "ctrl_g12_best",
            "source_checkpoint_sha256": "b" * 64,
            "coef": dict.fromkeys(("centroid", "logsigma", "shape"), 1),
            **hashes,
        },
    )
    hashes["eval_coef_sha256"] = sha256_file(eval_coef)
    baseline = save(
        "baseline.json",
        {
            "schema": SCHEMA,
            "purpose": "s_baseline",
            "source_epoch": 50,
            "source_role": "baseline_vggfix_ep50",
            "source_checkpoint_sha256": "c" * 64,
            "L1_JT": 1,
            "L1_BF": 2,
            "J": 3,
            **hashes,
            "legacy_l1_reconciliation": {"status": "passed"},
        },
    )
    hashes["s_baseline_sha256"] = sha256_file(baseline)
    cal = save("calibration.json", {**calibration(), **hashes})
    save("warm.pth.json", {"status": "validated", **hashes})
    blind = save(
        "blind.json",
        {
            "schema": SCHEMA,
            "purpose": "fixed_blind_manifest",
            **hashes,
        },
    )
    return Namespace(
        dual_calibration_json=cal,
        dual_eval_calibration_json=eval_coef,
        dual_target_cache=str(tmp_path / "cache"),
        dual_sample_manifest=sample,
        finetune=warm,
        dual_parent_checkpoint=parent,
        fixed_pair_path=fixed,
        s_baseline_path=baseline,
        json_path=[train],
        val_json_path=[val],
        dual_blind_manifest=blind,
        output_dir=str(tmp_path / "output"),
        no_gan=True,
        freeze_encoder=False,
        fusion_lambda=0.5,
        half_mask_ratio=0.0,
        jieti_pool=224,
        jieti_k_max=4,
        jieti_alpha_jt=1,
        jieti_soft_fg="linear",
        jieti_pred_mass_ratio=0.1,
        jieti_valid_ink=200,
        num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1,
        style_weight=14.73,
        clip_grad=3.0,
        drop_path=0.1,
        layer_decay=0.8,
        augmentation_policy="finetune",
        seed=0,
        model="vit_base_patch16_input896x448_win_dec64_8glb_sl1",
        num_mask_patches=784,
        max_mask_patches_per_block=392,
        edge_warmup_epochs=10,
        loss_warmup_duration=8,
        edge_weight_final=0.2,
        warmup_epochs=10,
        lr=0.00195699,
        min_lr=0.0,
        save_best=True,
        save_best_only=False,
        save_freq=10,
        input_size=[896, 448],
        batch_size=2,
        accum_iter=8,
        world_size=4,
        auto_resume=False,
        resume="",
        enable_deepspeed=False,
    )


def test_preflight_actual_files_reject_resume_and_old_output(tmp_path):
    from pathlib import Path

    from tools.dual_view_protocol import validate_training_inputs

    args = training_fixture(tmp_path)
    assert validate_training_inputs(args)["status"] == "validated"
    assert len(args.dual_blind_manifest_sha256) == 64
    for name, bad in (
        ("resume", "old.pth"),
        ("enable_deepspeed", True),
        ("auto_resume", True),
        ("batch_size", 1),
        ("half_mask_ratio", 0.1),
        ("save_best_only", True),
    ):
        modified = copy.deepcopy(args)
        setattr(modified, name, bad)
        with pytest.raises(ValueError):
            validate_training_inputs(modified)
    output = Path(args.output_dir)
    output.mkdir()
    (output / "checkpoint-0.pth").write_bytes(b"old result")
    with pytest.raises(ValueError, match="previous results"):
        validate_training_inputs(args)


def test_training_preflight_rejects_real_input_hash_change(tmp_path):
    from pathlib import Path

    from tools.dual_view_protocol import validate_training_inputs

    args = training_fixture(tmp_path)
    Path(args.json_path[0]).write_text('[{"new": true}]')
    with pytest.raises(ValueError, match="train_manifest_sha256"):
        validate_training_inputs(args)


@pytest.mark.parametrize(
    "replacement",
    [
        {"status": "unvalidated"},
        {"coef": {}},
        {"coef": {"centroid": True, "logsigma": 1, "shape": 1}},
    ],
)
def test_preflight_rejects_unvalidated_or_missing_eval_coef(
    tmp_path, replacement
):
    from pathlib import Path

    from tools.dual_view_protocol import check_eval_inputs, current_hashes

    args = training_fixture(tmp_path)
    mapped = Namespace(
        json_path=args.json_path,
        val_json_path=args.val_json_path,
        target_cache=args.dual_target_cache,
        samples=args.dual_sample_manifest,
        checkpoint=args.finetune,
        parent_checkpoint=args.dual_parent_checkpoint,
        fixed_pair_path=args.fixed_pair_path,
        eval_coef=args.dual_eval_calibration_json,
        s_baseline=args.s_baseline_path,
    )
    path = Path(args.dual_eval_calibration_json)
    path.write_text(
        json.dumps({**json.loads(path.read_text()), **replacement})
    )
    with pytest.raises(ValueError):
        check_eval_inputs(mapped, current_hashes(mapped))


def test_forward_probe_matches_engine_bfloat16_inputs_labels_untouched():
    torch = pytest.importorskip("torch")
    from tools.dual_view_protocol import TERMS, forward_terms

    class Model:
        def __call__(self, image, target, mask, valid, **kwargs):
            assert all(
                value.dtype == torch.bfloat16
                for value in (image, target, mask, valid)
            )
            labels = kwargs["dual_targets"]
            assert labels["has_jt"].dtype == torch.bool
            assert labels["instance_classes"].dtype == torch.long
            return ({term: image.sum() * 0 for term in TERMS},)

    batch = {
        "image": torch.ones(2, 3, 4, 2),
        "target": torch.ones(2, 3, 4, 2),
        "mask": torch.zeros(2, 2, 2),
        "valid": torch.ones(2, 3, 4, 2),
        "has_jt": torch.tensor([True, False]),
        "instance_classes": torch.zeros(2, 4, dtype=torch.long),
    }
    assert set(forward_terms(Model(), batch)) == set(TERMS)


@pytest.mark.parametrize('ratio', [0.5, 0.5001025512, 0.4996875285])
def test_ratio_accepts_measured_bfloat16_backward_variation(ratio):
    from tools.dual_view_protocol import check_gradient_ratio

    check_gradient_ratio(ratio, 0.5, 'j_base_ratio')


@pytest.mark.parametrize('ratio', [None, 0, float('nan'), float('inf'),
                                  0.49, 0.502, True])
def test_ratio_still_rejects_missing_degenerate_or_wrong_balance(ratio):
    from tools.dual_view_protocol import check_gradient_ratio

    with pytest.raises(ValueError):
        check_gradient_ratio(ratio, 0.5, 'j_base_ratio')


def test_minimum_ratio_preserves_full_branch_lower_bound():
    from tools.dual_view_protocol import check_gradient_ratio

    check_gradient_ratio(0.7, 0.5, 'full_pen', minimum=True)
    check_gradient_ratio(0.4996875285, 0.5, 'full_pen', minimum=True)
    with pytest.raises(ValueError):
        check_gradient_ratio(0.49, 0.5, 'full_pen', minimum=True)


def test_measurement_stage_is_preserved_without_validated_output(tmp_path):
    from tools.dual_view_protocol import save_calibration_stage

    output = str(tmp_path / 'train_coef.json')
    records = [{'grams': {'vit': [[1, 0], [0, 1]]}}]
    path = save_calibration_stage(output, 'final', records,
                                  {'sample_manifest_sha256': 'a' * 64},
                                  {'w_j': 0.2})
    saved = json.loads((tmp_path / 'train_coef.json.final.json').read_text())
    assert saved['status'] == 'measured'
    assert saved['measurements'] == records
    assert not (tmp_path / 'train_coef.json').exists()
    with pytest.raises(FileExistsError):
        save_calibration_stage(output, 'final', records, {}, {})
    assert path.endswith('.final.json')
