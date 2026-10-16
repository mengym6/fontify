Run B Fix Q spatial probe
========================

This is stage A of the user's 2026-10-16 experimental plan. No training is
performed. Stage B remains a proposal and is not launched automatically.

Source: original-edge Run B Fix best epoch 50, SHA256
57ac8276e3468ad92ce2aff033736128b33862a4f922618c51910717166ae852.
Tip and G Scale checkpoints are rejected. The default model path is unchanged.

Production tensors are four lower taps concatenated as (1,28,28,3072),
Q (1,784,256), four occupancy maps (1,4,28,28), and a layout tensor
(1,28,28,532) ordered G(276), Q(256). The fusion is joint LayerNorm followed
by Linear(532,3072). This probe explicitly freezes the complementary slice
AFTER normalization: A1 changes normalized Q only; A2 changes raw occupancy,
recomputes normalized G only, and restores the original normalized Q.
A3 changes the lower taps only and preserves the entire original residual.
A4 combines A1/A3; A5 shifts the support to an equal-area wrong region.
AO warps Q in GT occupancy regions and is always marked privileged.

All axes use cell-centre inverse bilinear resampling with align_corners=False,
zero padding and soft removal/deposition. Contributions with overlap are
normalized. Vacated features tend towards zero; this is a sensitivity probe,
not an assumed correct source-to-target correspondence. A5 uses deterministic
half-grid toroidal shifts (and therefore can wrap at a boundary); its support
geometry is measured independently. Identity ratio bypasses sampling exactly.

The same raw soft occupancy is used for every predicted-slot intervention.
Existence >=0.5 and soft area >=1 grid cell determine eligibility without GT.
All four slots, failures and post-hoc GT matches are logged. BF has no GT
instance metrics. Per the stage A contract, only JT instances are intervened
on. BF predictions are retained as no-ops in complete-240 scoring; identical
BF quality is by design, not evidence of robustness after training.

Usage, in the server's fontify environment::

  python -u tools/probe_q_spatial_control.py \
    --checkpoint models/finetune_dual_view_runb_fix_20261015/checkpoint-best.pth \
    --reference-eval models/dual_runb_fix_train_20261015/best_eval.json \
    --prep models/dual_view_pad_20261014 \
    --candidate-summary outputs/layout_diagnostic_fix_ep50_20261015/summary.json \
    --output outputs/q_spatial_layout_runb_fix_20261016

The launcher runs independent shards on GPUs 0 and 1 and aggregates only
after both succeed. --limit 2 runs a disposable real-model validation of the
first two BF items. The performed mixed BF/JT smoke instead used worker rank0,
shards14, limit15 (indices0 and14), with a precreated temporary figures folder.
Each A0 output is compared bitwise with predict_full, and per-item L1/J
reconcile to the sealed evaluation at rtol=1e-6, atol=1e-7. Every identity
condition is pixel-exact. The sealed evaluator historically allowed cuDNN
TF32: a separate native prediction in that original backend reconciles every
sealed L1/J value. The entire paired A0--AO experiment disables both TF32
flags, and reports the strict/original A0 differences explicitly. No sealed
reconciliation tolerance is relaxed. GT perturbation, fixed upstream outputs, upper
tap identity, complete parameter digests and checkpoint SHA are checked.

Final deliverables include per-item/per-instance CSVs, slot selection logs,
condition L1/J/S, response quantiles, all condition montages, blinded paired
images, unfilled human scores, provenance and SHA256. Worker logs/shards
are removed only after successful aggregation. No feature cache or checkpoint
is written. Smoke output is disposable, never scientific evidence.

Main response tables use actual support sigma changes >=20%, separate
enlargement/shrinkage and directions, and include the old 20 candidates as
an exploratory subgroup. Geometry uses fixed GT Voronoi and 224px soft ink,
converted to 448px units, with fixed ROI/hard-threshold sensitivity checks.
J and S reuse the original sealed evaluation contract. Quality proxies
(224px thin/short-stroke recall, components, endpoints, extra ink, highpass)
do not replace blind manual tip/ghosting/break assessment. No automatic
quality pass or independent generalization claim is made from this old val.
