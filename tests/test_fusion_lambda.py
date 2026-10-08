"""训练时融合权重 --fusion_lambda 的单测（无 torch 环境可直接 python 运行）。

- main_train：参数默认 0.5、建模型后设到 model.fusion_lambda，有 (0,1) 校验；
- eval_s_baseline.load_model：从 ckpt['args'] 读 fusion_lambda，旧 ckpt 无此字段按 0.5；
- export_fusion_trained：(λ, ckpt) 对的解析、列头、λ 核对；
- 新训练脚本与 ctrl_g12.sh 只差约定的几项。
"""

import argparse
import ast
import copy
import difflib
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.export_fusion_trained as ft  # noqa: E402


def _func(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _exec_func(path, name, ns):
    node = _func(path, name)
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), str(ROOT / path), "exec"), ns)
    return ns[name]


# ---------------------------------------------------------------------------
# main_train
# ---------------------------------------------------------------------------

def _parse(argv):
    get_args_parser = _exec_func("main_train.py", "get_args_parser", {"argparse": argparse})
    old = sys.argv
    try:
        sys.argv = ["main_train.py"] + argv
        args, _ = get_args_parser()
    finally:
        sys.argv = old
    return args


def test_main_train_fusion_lambda_arg_default_and_value():
    assert _parse([]).fusion_lambda == 0.5
    assert isinstance(_parse([]).fusion_lambda, float)
    assert _parse(["--fusion_lambda", "0.3"]).fusion_lambda == 0.3


def test_main_train_sets_model_attr_and_validates():
    src = (ROOT / "main_train.py").read_text(encoding="utf-8")
    main = ast.get_source_segment(src, _func("main_train.py", "main"))
    build = main.index("model = models_train.__dict__[args.model]()")
    check = main.index("if not 0.0 < args.fusion_lambda < 1.0:")
    setattr_ = main.index("model.fusion_lambda = args.fusion_lambda")
    # 建模型之后、DDP 包装之前设属性（DDP 之后 model 换成外壳，属性要设在内层模块上）
    ddp = main.index("torch.nn.parallel.DistributedDataParallel(model")
    assert build < check < setattr_ < ddp
    # 校验逻辑与 main_train 的写法一致：(0,1) 开区间
    ok = lambda lam: 0.0 < lam < 1.0  # noqa: E731
    assert ok(0.5) and ok(0.3) and ok(0.7)
    assert not any(ok(v) for v in (0.0, 1.0, -0.1, 1.2))


# ---------------------------------------------------------------------------
# eval_s_baseline.load_model
# ---------------------------------------------------------------------------

class _Model:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {}

    def load_state_dict(self, sd, strict=True):
        self.loaded = sd


def _load_model_with(ckpt):
    fake_torch = types.SimpleNamespace(load=lambda path, map_location=None: ckpt)
    fake_mt = types.SimpleNamespace(vit_base_patch16_input896x448_win_dec64_8glb_sl1=_Model)
    load_model = _exec_func("tools/eval_s_baseline.py", "load_model",
                            {"torch": fake_torch, "models_train": fake_mt})
    return load_model("x.pth")


def test_load_model_reads_lambda_from_ckpt_args():
    for lam in (0.3, 0.7, 0.5):
        m = _load_model_with({"model": {"w": 1}, "args": argparse.Namespace(fusion_lambda=lam)})
        assert m.fusion_lambda == lam and m.loaded == {"w": 1}


def test_load_model_old_ckpt_defaults_to_half():
    # 旧 ckpt：args 里没有 fusion_lambda（ctrl_g12、baseline）或根本没有 args（预训练权重）
    for ckpt in ({"model": {}, "args": argparse.Namespace(lr=1.0)}, {"model": {}}):
        m = _load_model_with(ckpt)
        assert m.fusion_lambda == 0.5
    # 0.5 走 fuse_streams 的原式分支，旧 ckpt 行为不变
    from util.fusion import fuse_streams
    import numpy as np
    x = np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32)
    assert np.array_equal(fuse_streams(x, m.fusion_lambda), (x[:2] + x[2:]) * 0.5)


# ---------------------------------------------------------------------------
# export_fusion_trained
# ---------------------------------------------------------------------------

def test_parse_pairs_sorted_and_checked():
    pairs = ft.parse_pairs([["0.7", "c"], ["0.3", "a"], ["0.5", "b"]])
    assert pairs == [(0.3, "a"), (0.5, "b"), (0.7, "c")]
    assert ft.column_heads([p[0] for p in pairs]) == [
        "ref (upper GT)", "GT (lower)", "λ=0.3 (trained)", "λ=0.5 (trained)", "λ=0.7 (trained)"]
    for bad in ([["0.3", "a"], ["0.7", "c"]],           # 缺 0.5
                [["0.5", "a"], ["0.5", "b"]],           # 重复
                [["0.0", "a"], ["0.5", "b"]],           # 训练时 λ 须在开区间
                [["0.5", "a"], ["1.0", "b"]]):
        try:
            ft.parse_pairs(bad)
            raise AssertionError(f"{bad} 应报错")
        except ValueError:
            pass


def test_check_ckpt_lambda():
    ft.check_ckpt_lambda(0.3, 0.3, "a")
    ft.check_ckpt_lambda(0.3, 0.30000000000000004, "a")
    try:
        ft.check_ckpt_lambda(0.3, 0.5, "a")
        raise AssertionError("λ 不一致应报错")
    except ValueError as e:
        assert "a" in str(e)


# ---------------------------------------------------------------------------
# 训练脚本
# ---------------------------------------------------------------------------

def _changed(a, b):
    """逐行 diff，返回 (删去的行, 新增的行)，忽略新增的说明注释。"""
    la = a.splitlines()
    lb = b.splitlines()
    rem, add = [], []
    for line in difflib.ndiff(la, lb):
        if line.startswith("- "):
            rem.append(line[2:])
        elif line.startswith("+ ") and not line[2:].startswith("#"):
            add.append(line[2:])
    return rem, add


def test_lam_scripts_only_differ_in_agreed_items():
    base = (ROOT / "finetune_jieti_ctrl_g12.sh").read_text(encoding="utf-8")
    for tag, lam, port in (("lam03", "0.3", "29556"), ("lam07", "0.7", "29557")):
        new = (ROOT / f"finetune_jieti_ctrl_g12_{tag}.sh").read_text(encoding="utf-8")
        rem, add = _changed(base, new)
        assert sorted(rem) == sorted([
            "name=finetune_jieti_ctrl_g12",
            "MASTER_PORT=${MASTER_PORT:-29555}",
            "python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT \\",
            "    --accum_iter 8  \\",
            "    --s_baseline_path models/jieti_search/s_baseline.json \\",
            "    --grad_log_interval 5",
        ])
        assert sorted(add) == sorted([
            f"name=finetune_jieti_ctrl_g12_{tag}",
            "MASTER_PORT=${MASTER_PORT:-%s}" % port,
            "python -m torch.distributed.launch --nproc_per_node=2 --master_port=$MASTER_PORT \\",
            "    --accum_iter 16  \\",
            "    --s_baseline_path models/jieti_search2/s_baseline.json \\",
            "    --grad_log_interval 5 \\",
            f"    --fusion_lambda {lam}",
        ])
        # 有效 batch 不变：batch 2 × accum × nproc
        bs = int(re.search(r"--batch_size (\d+)", new).group(1))
        acc = int(re.search(r"--accum_iter (\d+)", new).group(1))
        npr = int(re.search(r"--nproc_per_node=(\d+)", new).group(1))
        assert bs * acc * npr == 64


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
