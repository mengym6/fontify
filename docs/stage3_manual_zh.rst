阶段 3 操作手册
================

适用范围
--------

本手册对应仓库 ``3df3d1d`` 之后的阶段 3 代码（``tools/stage3.py``、
``tools/stage3_experiments.py``、``fontdata_example/stage3_json``）。数据契约和
设计说明见 ``docs/stage3_zh.rst``，本手册只讲"怎么跑、看什么、怎么定"。

所有命令在服务器仓库根目录、CUDA/detectron2 训练环境下执行。每个 ``--output``
必须是不存在的目录；训练不会自动续跑。``runs/`` 已在 ``.gitignore`` 中。

阶段 3 的目标是用控制变量对照回答三个问题：structure 四个子项按什么比例组合、
structure/detail 总权重多大（或为 0）、参考风格模块是否有效。程序不会自动选赢家，
以下标注 **【决定】** 的地方必须由人判断并记录。

流程总览
--------

::

    准备 → audit → smoke → evaluate×N【决定 1：初始化】 → fit32 → calibrate
      → internal【决定 2：系数】 → structure【决定 3：W_S】 → detail【决定 4：W_D】
      → style【决定 5：风格模块】 → local【决定 6：最终值】 → 种子复现【决定 7：是否确认】
      → 登记 AGENTS.md

串行依赖只在决策点之间；一轮内部各组彼此独立，可并行。总预算约 14 组 400-update
训练加 4 组种子复现，每组样本量约等于 32 个 epoch。

阶段 3 实际遮盖设置（登记时按此描述）：带语义 mask 的 CalliPhase 样本全部使用语义遮盖，
JT 抽 1 层、BF 抽 11 层，不使用随机块和半遮盖；入口参数 0.8/0/0.2 不生效。

第 0 步：准备
-------------

::

    git pull && git log --oneline -1          # 不早于 3df3d1d
    PYTHONPATH=. python -m pytest tests/test_stage3.py -q   # 期望 15 passed
    python -c "import tensorboard"            # 训练默认写 TB，缺包会直接报错
    ls vgg19/vgg19-dcbb9e9d.pth models/vit_base_font/checkpoint-14.pth
    ls fontdata_example/stage3_json/manifest.json
    ls models/finetune_stele_loss_diag_exp*/   # 阶段 2 候选 checkpoint，记下路径

以下变量后续命令都会用到，每次登录后先设置::

    MANIFEST=fontdata_example/stage3_json/manifest.json
    DATA=fontdata_example
    MASKS=fontdata_example/font/train/new

第 1 步：audit
--------------

::

    python tools/stage3.py audit --manifest $MANIFEST --data-root $DATA \
      --output runs/stage3/audit

看 ``runs/stage3/audit/audit.json``：

- ``train.records`` = 1594，``val_seen.records`` = 286，两者 ``styles`` = 9；
- ``image_and_metadata_sha256`` 以 ``5af150dd`` 开头。

不一致就停：说明服务器上的图片或 JSON 与本机不同，先核对数据再继续。

第 2 步：smoke
--------------

::

    python tools/stage3.py smoke --manifest $MANIFEST --data-root $DATA \
      --checkpoint models/vit_base_font/checkpoint-14.pth \
      --style-mode reference --output runs/stage3/smoke

看 ``runs/stage3/smoke/smoke.json``：``passed: true``、``loss`` 与
``pre_clip_grad_norm`` 为有限数、``checkpoint_roundtrip: true``。

这是阶段 3 代码第一次在完整 ViT 上运行。任何报错都保留完整 traceback，不要改阈值绕过。

第 3 步：选初始化【决定 1】
---------------------------

对阶段 1 的 ``checkpoint-14`` 和每个阶段 2 候选跑同一套评估::

    for c in models/vit_base_font/checkpoint-14.pth \
             models/finetune_stele_loss_diag_exp*/checkpoint-34.pth; do
      python tools/stage3.py evaluate --manifest $MANIFEST --data-root $DATA \
        --checkpoint "$c" \
        --output runs/stage3/select/$(basename $(dirname "$c"))-$(basename "$c" .pth)
    done
    python tools/stage3.py summarize --metrics runs/stage3/select/*/metrics.json \
      --output runs/stage3/select/summary.json

epoch 号按实际存在的文件改。每个目录含 ``metrics.json`` 和条带图，条带图三联为
参考字 | 预测 | GT，文件名 ``val_seen-0000-correct.png`` 等。

看什么：

1. 条带图（先看图再看数）：笔锋是否糊成团、结体是否松散偏移、有无彩色/灰底伪影。
2. ``summary.json`` → ``runs[]`` 中 ``split: val_seen`` 的 ``mean``：指标含义见附录 B。
3. ``metrics.json`` 中 ``case`` 为 ``blank`` / ``other_style`` 的行的
   ``reference_output_delta``：接近 0 表示模型不看参考字。

定什么：一个 checkpoint，记为 ``$SELECTED``。不要按训练 loss 最低选。

记什么：候选列表、各自 val_seen 均值、选择及理由。

第 4 步：fit32
--------------

::

    SELECTED=models/xxx/checkpoint-xx.pth
    torchrun --standalone --nproc_per_node=2 tools/stage3.py train \
      --manifest $MANIFEST --data-root $DATA --checkpoint "$SELECTED" \
      --fit32 --updates 300 --output runs/stage3/fit32

单卡：去掉 ``torchrun --standalone --nproc_per_node=2``，改为
``python tools/stage3.py train ... --accum-iter 64``（代码要求
batch × accum × 卡数 = 128）。

看什么（``tensorboard --logdir runs/stage3``，或 ``train.jsonl``）：

- ``train/loss_rank0`` 在 300 update 内应明显下降并趋平（这是 32 个固定样本的
  过拟合测试，降不下去说明优化链路有问题）；
- ``train/grad_post_clip`` ≤ 3.0，``train/grad_pre_clip`` 有限且不持续暴涨；
- ``update_norm/blocks.9`` … ``blocks.11``、``decoder*`` 非零；``blocks.0`` … ``blocks.8``
  与 ``patch_embed`` 应为 0（冻结）；
- ``eval-300/train-*-correct.png`` 应接近 GT。

不通过就停，把 ``train.jsonl`` 末尾 20 行和 TB 截图一起分析。

第 5 步：calibrate
------------------

::

    python tools/stage3.py calibrate --manifest $MANIFEST --data-root $DATA \
      --checkpoint "$SELECTED" --output runs/stage3/calibration

看 ``runs/stage3/calibration/calibration.json``：

- ``status`` 必须是 ``ok``。``blocked`` 时 ``reason`` 为
  ``Required coefficient outside bounds``（系数超出 [0.1, 100]）、
  ``Zero/nonfinite pixel gradient`` 或 ``Degenerate parameter gradient``；
  **不要放宽阈值**，先分析原因。
- ``structure_coefficients``：row / col / centroid / area 四个系数。预期 row、col
  远大于 1，area 远小于 1（阶段 2 测得 row/col 梯度约 5e-4、area 约 1.6）。
- ``structure_common_scale``：公共倍率，使合成参数梯度中位数与标定前一致。
- ``measurements[].parameter_cosines``：4×4 子项参数梯度余弦。centroid 与 area
  若持续为负，说明两项方向相反、标定后可能互相抵消，写进结论。

第 6 步：五轮扫描
-----------------

每轮套路相同：生成命令 → 执行 → summarize → 看图看指标 → 定值 → 记录。
每轮都有一组与上一轮赢家配置完全相同（同 checkpoint、权重、系数、seed、400 update），
不重跑，把旧目录的 ``metrics.json`` 一并传给 summarize。

一次性定义（每次登录后执行）::

    SELECTED=models/xxx/checkpoint-xx.pth
    COEF=runs/stage3/calibration/calibration.json
    GEN="python tools/stage3_experiments.py"
    COMMON="--manifest $MANIFEST --data-root $DATA --checkpoint $SELECTED \
      --semantic-mask-dir $MASKS --output-root runs/stage3"

    # 顺序执行一份 commands.json。SKIP=名字1,名字2 跳过重复组；GPUS=1 改为单卡命令。
    run_commands () {
    python - "$1" <<'EOF'
    import json, os, subprocess, sys
    skip = set(filter(None, os.environ.get("SKIP", "").split(",")))
    for c in json.load(open(sys.argv[1])):
        if c["name"] in skip:
            print("skip", c["name"]); continue
        argv = c["argv"]
        if os.environ.get("GPUS") == "1":
            argv = ["python"] + argv[3:] + ["--accum-iter", "64"]
        print("==>", c["name"], "seed", c["seed"], flush=True)
        subprocess.run(argv, check=True)
    EOF
    }

每组输出目录 ``runs/stage3/<轮>/<组名>-seed-0/``，内含 ``run.json``、``train.jsonl``、
``tensorboard/``、``checkpoint-update-{50..400}.pth``、``eval-{50..400}/``。
``eval-N/`` 同时包含 train 和 val_seen 两个划分，可直接比较训练/验证差距。

判读规则（每轮通用）：

1. 先盲看图：把各组 ``eval-400/val_seen-*-correct.png`` 混在一起看，不看目录名；
2. 再看 ``summary.json`` 的 val_seen 均值，同时看 train 与 val_seen 的差距，
   差距突然拉大即过拟合；
3. 差异小于种子间波动（第 7 步才能知道）的暂不下结论；
4. 允许的结论包括"权重 0 最好"或"风格模块无效"，不要为了有结果而选非零。

轮 1：internal【决定 2：子项系数】
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

两组：``equal``（系数全 1）与 ``calibrated``（标定系数），其余均为
structure 0.2、detail 0.2、风格模块 off。这是共同对照值，不代表最优。 ::

    $GEN internal $COMMON --coefficients $COEF \
      --output-json runs/stage3/internal-commands.json
    run_commands runs/stage3/internal-commands.json
    python tools/stage3.py summarize \
      --metrics runs/stage3/internal/*/eval-400/metrics.json \
      --output runs/stage3/internal/summary.json

看：val_seen 的 ``row``、``col``、``centroid``、``area``（标定针对的四项），
以及 ``highpass``、``edge_f1`` 有没有被拖坏。

定并写入变量（后续所有轮沿用）::

    COEF_ARG="--coefficients $COEF"; WINNER=runs/stage3/internal/calibrated-seed-0   # 标定胜出
    # COEF_ARG=""; WINNER=runs/stage3/internal/equal-seed-0                          # 等系数胜出

轮 2：structure【决定 3：W_S】
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

三组：structure 总权重 0 / 0.2 / 0.5，detail 默认固定 0.2。
``structure-0.2`` 与轮 1 赢家相同，配置完全一致时跳过。 ::

    $GEN structure $COMMON $COEF_ARG --output-json runs/stage3/structure-commands.json
    SKIP=structure-0.2 run_commands runs/stage3/structure-commands.json
    python tools/stage3.py summarize \
      --metrics runs/stage3/structure/*/eval-400/metrics.json $WINNER/eval-400/metrics.json \
      --output runs/stage3/structure/summary.json

看：``structure-0`` 是关键对照，若它与其他组的结体指标（``centroid``、``row``、
``col``、``area``、``aspect``）差不多，说明结构损失当前无效，选 0 合法。

定::

    W_S=0.2        # 按实际结果选择，写法与目录名一致：0、0.2、0.5

轮 3：detail【决定 4：W_D】
~~~~~~~~~~~~~~~~~~~~~~~~~~~

三组：detail 0 / 0.2 / 0.5，structure 固定 ``$W_S``。
``detail-0.2`` 与轮 2 的 ``structure-$W_S`` 组相同，配置完全一致时跳过。 ::

    $GEN detail $COMMON $COEF_ARG --structure-weight $W_S \
      --output-json runs/stage3/detail-commands.json
    SKIP=detail-0.2 run_commands runs/stage3/detail-commands.json
    PREV=runs/stage3/structure/structure-$W_S-seed-0   # 若 W_S=0.2 则 PREV=$WINNER
    python tools/stage3.py summarize \
      --metrics runs/stage3/detail/*/eval-400/metrics.json $PREV/eval-400/metrics.json \
      --output runs/stage3/detail/summary.json

看：``highpass``、``gradient``、``edge_f1`` 随权重的变化，条带图上笔锋是否更实；
同时看 ``area``，detail 权重过大常让笔画变粗。

定::

    W_D=0.2        # 按实际结果选择：0、0.2、0.5

轮 4：style【决定 5：风格模块】
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

三组：off / reference / constant，权重固定 ``$W_S`` / ``$W_D``。
``off`` 与轮 3 的 ``detail-$W_D`` 组相同，跳过。 ::

    $GEN style $COMMON $COEF_ARG --structure-weight $W_S --detail-weight $W_D \
      --output-json runs/stage3/style-commands.json
    SKIP=off run_commands runs/stage3/style-commands.json
    OFF=runs/stage3/detail/detail-$W_D-seed-0   # 若 W_D=0.2 则 OFF=$PREV
    python tools/stage3.py summarize \
      --metrics runs/stage3/style/*/eval-400/metrics.json $OFF/eval-400/metrics.json \
      --output runs/stage3/style/summary.json

看（三者必须一起读）：

==================================  ==========================================
结果                                结论
==================================  ==========================================
reference 优于 off 且优于 constant  模块确实在读参考字
reference ≈ constant，均优于 off    收益来自多了可训练参数，不是参考信息
reference ≈ off                     模块无效
==================================  ==========================================

另看 ``metrics.json`` 中 ``blank`` / ``other_style`` 行的 ``reference_output_delta``：
reference 组应明显大于 off 组，否则模块没在用参考。

定：风格模块是否有效（决定最终模型选 reference 还是 off）。

轮 5：local【决定 6：最终值】
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 reference 模型上每次只把一个损失 ×0.5 或 ×2；权重为 0 的损失不生成扰动组。
``reference`` 与轮 4 的 reference 组相同，跳过。 ::

    $GEN local $COMMON $COEF_ARG --structure-weight $W_S --detail-weight $W_D \
      --output-json runs/stage3/local-commands.json
    SKIP=reference run_commands runs/stage3/local-commands.json
    python tools/stage3.py summarize \
      --metrics runs/stage3/local/*/eval-400/metrics.json \
                runs/stage3/style/reference-seed-0/eval-400/metrics.json \
                $OFF/eval-400/metrics.json \
      --output runs/stage3/local/summary.json

看：×2 或 ×0.5 是否明显优于中心点；对照里始终带着 ``$OFF``（无风格模块基线）。

定：最终 ``W_S``、``W_D``、style 模式，以及最终候选目录 ``$CANDIDATE``。

第 7 步：种子复现【决定 7】
---------------------------

对最终候选和它的基线（style=off、同权重）各补种子 1、2。用 ``local`` 生成后只执行
需要的两组（其余用 SKIP 跳过，组名见 ``local-commands.json``）::

    $GEN local $COMMON $COEF_ARG --structure-weight $W_S --detail-weight $W_D \
      --seeds 1 2 --output-json runs/stage3/seeds-commands.json
    SKIP=structure-x0.5,structure-x2,detail-x0.5,detail-x2 \
      run_commands runs/stage3/seeds-commands.json
    # 基线（off）种子 1、2 需手动生成：把 style 轮的 off 组 argv 改 --seed 与 --output 后执行
    python tools/stage3.py summarize \
      --metrics $CANDIDATE/eval-400/metrics.json runs/stage3/local/reference-seed-{1,2}/eval-400/metrics.json \
                $OFF/eval-400/metrics.json <基线 seed1/seed2 的 metrics.json> \
      --output runs/stage3/final-summary.json

看 ``final-summary.json`` → ``across_seeds[]``：同一 ``identity`` 下三个种子的 ``mean``
与 ``seed_std``。候选与基线的均值差 **大于** 各自 ``seed_std`` 才算确认；否则写
"未确认"，阶段 3 到此为止，不再扩展实验。

第 8 步：登记
-------------

在 ``AGENTS.md`` 阶段 3 段落按模板补齐，仅记录真实运行：

- 每轮命令与 seed（``*-commands.json`` 可直接引用）；
- ``run.json`` 的 ``data_sha256`` 与 ``manifest_sha256``；
- ``$SELECTED``、最终 checkpoint 路径 ``<组目录>/checkpoint-update-400.pth``；
- 固定验证图（``eval-400/val_seen-*-correct.png``）、指标表、train/val 差距；
- 人工结论；阶段表状态改为"已完成（日期、checkpoint 路径）"。

附录 A：输出文件与字段
----------------------

``audit.json``
    ``train`` / ``val_seen`` 各含 ``records``、``styles``；``image_and_metadata_sha256``
    为图片内容 + JSON 记录的联合指纹；``semantic_masks_hashed`` 恒为 false（语义 mask
    版本需单独保存，mask 改动后不得复用旧基线）。

``smoke.json``
    ``passed``、``device``、``loss``、``pre_clip_grad_norm``、``checkpoint_roundtrip``。

``run.json``（每个 train/evaluate/calibrate 目录）
    ``args``（含 seed、checkpoint、updates）、``config``（style_mode、权重、系数、
    vgg_input_mode 等）、``manifest_sha256``、``data_sha256``。summarize 用它自动
    判断哪些目录属于同一配置。

``train.jsonl``（每行一个 update）
    ``update``、``loss_rank0``（rank 0 上累积均值，不是全局均值）、``grad_pre_clip``、
    ``grad_post_clip``、``last_microbatch_losses``（**最后一个 micro-batch** 的值，
    非均值；键包括 ``structure`` 及 ``structure_row/col/centroid/area``、
    ``structure_weight``、``structure_weighted``、``detail``、``highpass``、
    ``gradient``、``detail_weight``、``detail_weighted``、``gradient_contribution``、
    ``detail_region_ratio``、``valid_ratio``、``target_highpass_mean``、
    ``target_gradient_mean``、各 ``*_grad_ok``）。recon/style/edge 不在其中，只体现在
    ``loss_rank0``。每 5 个 update 附 ``parameters``：按参数组给出 ``total``、
    ``trainable``、``learning_rates``、``gradient_norm``、``update_norm``。

``tensorboard/``
    与 train.jsonl 同源：``train/*``、``lr/max``、``last_microbatch_losses/*``、
    ``gradient_norm/<组>``、``update_norm/<组>``、``eval_<split>/<指标>``、
    ``eval_<split>/00..07``（条带图，可拖 step 看演变）。

``eval-N/metrics.json`` 与 ``evaluate`` 的 ``metrics.json``
    每行：``split``、``case``（correct / blank / same_style / other_style；训练中的
    eval 只有 correct）、``style_id``、``character``、``target``、``reference``、
    ``metrics``（8 项）、``reference_output_delta``（该 case 预测与 correct 预测的
    平均像素差）、``image``（条带文件名）。

``summary.json``
    ``runs[]``：每个 metrics 文件 × 划分的 ``count``、``mean``、``std``（只统计 correct）；
    ``across_seeds[]``：配置相同（checkpoint、config、数据 hash、update 数、eval 目录名）
    的目录按 seed 归并后的 ``mean`` 与 ``seed_std``。

``calibration.json``
    见第 5 步。

附录 B：评估指标
----------------

全部只在下半 query 区域、预测 vs GT 计算；除 ``edge_f1`` 越大越好外其余越小越好。
它们与训练损失定义不同（硬阈值、不乘 mask、不乘权重），不会因某个损失权重调大而
自动改善。

==============  =====================================================  ==========
指标            算法                                                   对应问题
==============  =====================================================  ==========
``highpass``    5×5 高斯高通后的平均绝对差                             笔锋、细节
``gradient``    Sobel 梯度图平均绝对差                                 边缘强度/方向
``edge_f1``     梯度幅值 > 0.1 二值化后逐像素 F1                       边缘位置
``centroid``    前景（灰度 < 0.9）质心的 \|Δx\| + \|Δy\|（坐标 [-1,1]） 字的位置
``row``         前景行投影的平均绝对差                                 结体（纵向分布）
``col``         前景列投影的平均绝对差                                 结体（横向分布）
``area``        前景占比之差                                           笔画粗细、墨量
``aspect``      前景外接框宽高比之差                                   整体比例
==============  =====================================================  ==========

分组读法：``highpass`` / ``gradient`` / ``edge_f1`` 一组看笔法；``centroid`` / ``row`` /
``col`` / ``area`` / ``aspect`` 一组看结体。一个改动通常只动其中一组，另一组不能变差。

附录 C：诊断表
--------------

按"看到什么 → 先查什么 → 可能原因 → 处理"排列。

**audit 阶段**

- hash 与本机不同 → ``git status``、图片目录 → 数据或 JSON 版本不一致 → 重新同步，
  不要在不一致的数据上继续。
- ``Cross-split target leakage`` → 报错中的路径 → 同一字的不同版本被拆到两侧 →
  用 ``build_stage3_manifest.py`` 重新生成，不要手改 JSON。

**smoke / fit32 阶段**

- ``Incompatible checkpoint`` → 缺失/多余键列表 → 候选 checkpoint 不是本仓库结构 →
  换 checkpoint；只允许缺 ``style_conditioner.*``。
- ``Nonfinite training loss`` → ``train.jsonl`` 最后一行 ``last_microbatch_losses`` →
  某分量为 inf/nan（多见于 ``structure_area`` 前景质量为 0） → 记录发生的 update 与
  样本，先在 CPU 上用该样本复现。
- fit32 的 loss 不降 → ``parameters`` 中各组 ``update_norm`` → 全为 0：参数被冻结或
  lr 为 0（看 ``learning_rates``）；非零但 loss 平：``grad_post_clip`` 是否长期 = 3.0
  （裁剪吃掉了全部梯度）。
- ``update_norm/blocks.0..8`` 非零 → 冻结失效 → 检查 ``freeze()`` 是否被调用（
  ``run.json`` 中 args 无相关开关，属代码问题）。

**calibrate 阶段**

- ``Required coefficient outside bounds`` → ``proposed_coefficients`` → 某子项像素梯度
  比其他小 100 倍以上（通常是 row/col） → 不放宽阈值；这本身就是结论：该子项在当前
  定义下无法通过系数拉平，写入登记并考虑 structure=0 或修改子项定义。
- ``Zero/nonfinite pixel gradient`` → ``measurements`` 中哪个 batch → 该 batch 预测
  前景为空 → 检查 ``$SELECTED`` 在 evaluate 中是否输出全白。

**训练阶段（各轮）**

- train 指标持续变好、val_seen 变差 → 对比 ``eval-50..400`` 的 val_seen 曲线 →
  过拟合 → 取 val_seen 最好的 update 的 checkpoint 作候选（不必是 400），并在登记中
  注明；不要用加大权重来"修"。
- 某组 ``highpass`` 变好但 ``area`` 明显变大 → 条带图看笔画是否变粗 → detail 权重
  过大 → 选更小的 ``W_D``。
- ``centroid`` / ``row`` / ``col`` 在 structure=0 和 0.5 之间几乎不变 → 看
  ``last_microbatch_losses`` 的 ``structure_weighted`` 相对 ``loss_rank0`` 的占比 →
  占比 < 1% 说明结构损失在总梯度里没有分量 → 结论"当前定义下无效"，不要继续加大。
- reference ≈ constant → ``update_norm/style_conditioner`` 是否非零 → 非零但无差别：
  模块学到的是与输入无关的偏置；为零：头部零初始化后梯度未传到 → 看 ``lr/max`` 与
  ``learning_rates`` 中是否有 3e-4。
- ``reference_output_delta`` 在 blank/other_style 上接近 0 → 模型忽略参考字 → 这是
  模型性质而非 bug；style 阶段的结论应据此写"未利用参考"。
- 同配置不同 seed 差异大于组间差异 → 本轮所有结论标"未确认"；可以增加 seed，但
  不要用单 seed 结果宣称有效。

**常见报错**

- ``File exists`` → 目录已存在（多为上次中途崩溃） → ``rm -rf`` 该组目录后重跑。
- ``batch_size * accum_iter * world_size must be 128`` → 单卡未加 ``--accum-iter 64``。
- ``ModuleNotFoundError: tensorboard`` → 安装，或加 ``--no-tensorboard``（不推荐，
  会丢失 TB 记录）。
- ``Calibration is not approved for use`` → 传入的 ``calibration.json`` 的 ``status``
  不是 ok → 不要改文件，回到第 5 步。
- ``No distinct-character reference for style/type`` → 某书家某 type 只有 1 个字 →
  当前 manifest 不会出现；若自行改过 JSON，重新生成。
