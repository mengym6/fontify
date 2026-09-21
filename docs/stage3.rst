Stage 3: reference conditioning and loss calibration
==================================================

Status (2026-09-20)
-------------------

Implementation and isolated CPU tests are available. Full-model CUDA smoke,
calibration, checkpoint selection, and stage-3 training have NOT been run.
The user supplies the selected stage-2 checkpoint and CalliPhase data.
Run every command from the repository root in the existing CUDA/detectron2
training environment. VGG19 weights are required at the existing repository
path. The temporary local test environment is not a replacement for that setup.

Stage 3 fixes detail to experiment 2: Gaussian highpass + 0.1 * Sobel,
kernel 5, sigma 1, batch-region normalization. There is no per-sample switch
in this entrypoint; a saved config requesting it is rejected.

Data contract
-------------

Supply a manifest, whose JSON paths are relative to the manifest itself::

    {
      "train": ["train.json"],
      "val_seen": ["val_seen.json"]
    }

Only these two splits exist; other keys are rejected. Every writer contributes
to both, so no unseen-writer metric is produced. Build the manifest from the
CalliPhase directory layout (``<writer>BF``/``<writer>JT`` folders with
``images_text_denoised`` and ``semantic_masks``)::

    python tools/build_stage3_manifest.py --data-root fontdata_example \
      --output fontdata_example/stage3_json

style_id is the writer name shared by its BF/JT folders; character is the first
character of the file stem; glyph_id is ``<writer>-<stem>``. The split unit is
(writer, character): all BF/JT entries and repeated writings such as ``永1`` of
one character land in the same split, and val_seen characters are held out
globally across writers. Per writer, about --val-ratio (default 0.15, the
historical train_json_new/val_json_new rule) of its characters go to val_seen,
never fewer than --min-val-characters. Targets without a source glyph or a
semantic mask are dropped and listed in split_summary.json.

Each referenced JSON is a list with the original image_path, target_path, type
fields, plus explicit style_id, character, glyph_id (all nonempty strings).
Example::

    {
      "image_path": "source/永.png",
      "target_path": "writer_a/images/永.png",
      "type": "BF",
      "style_id": "writer_a",
      "character": "永",
      "glyph_id": "writer_a-original-page17-char8",
      "source_dataset": "calliphase"
    }

Image paths are relative to --data-root. glyph_id identifies the original glyph
before cropping; all alternate crops retain this identity. A wrong glyph_id
cannot be detected reliably from pixels; audit combines supplied identities,
target file paths and exact file hashes. JT/BF duplication within a split is
allowed; cross-split target duplication is rejected. Shared source-font images
are allowed. This is not a guarantee of detecting arbitrary unlabelled near
duplicates.

val_seen characters must not occur in train, and every val_seen style must
occur in train. Every split needs distinct-character references per style; train also
needs them per type, and BF semantic sampling requires usable semantic sources.
Training reference pairing enforces both style identity and different character.
No test-set path is consumed by tuning commands. Keep a separate final test set.

Evaluation chooses references deterministically from the same evaluation split,
excluding the query character. Therefore it measures reference-conditioned
unseen-character generation, not a setting without access to style exemplars.
For complete sensitivity coverage, provide at least three characters per style
and at least two styles; absent alternate-reference cases are omitted explicitly
from metrics.json. Images contain reference / prediction / target without scores.

Preflight
---------

Use new output directories; existing directories are rejected, and training never
auto-resumes. Substitute real paths for the following examples::

    python tools/stage3.py audit --manifest /data/calli/manifest.json \
      --data-root /data/calli --output /runs/stage3-audit

    python tools/stage3.py smoke --manifest /data/calli/manifest.json \
      --data-root /data/calli --checkpoint /models/selected-stage2.pth \
      --style-mode reference --output /runs/stage3-smoke

    python tools/stage3.py evaluate --manifest /data/calli/manifest.json \
      --data-root /data/calli --checkpoint /models/selected-stage2.pth \
      --output /runs/stage2-reference-check

Repeat evaluation for stage-1 checkpoint-14 and other stage-2 candidates. Select
initialization from fixed validation images and metrics, not minimum training
loss. No command automatically declares a winner. Save the human decision.

smoke performs actual forward/backward, a clipped optimizer step, checkpoint save
and strict reload with prediction comparison. CPU unit tests do not replace it.

Check 32-sample fitting before expanding capacity::

    torchrun --standalone --nproc_per_node=2 tools/stage3.py train \
      --manifest /data/calli/manifest.json --data-root /data/calli \
      --checkpoint /models/selected-stage2.pth --fit32 --updates 300 \
      --output /runs/stage3-fit32

fit32 uses 32 fixed distinct targets, deterministic preprocessing and full query
masking. It uses no semantic augmentation. Parameter reports contain counts,
actual learning rates, post-clip gradient norms and update norms every 5 updates.
Loss components in train.jsonl are labelled last_microbatch_losses, not epoch
averages; loss_rank0 is the rank-0 accumulated mean, not a global DDP mean.

Structure calibration and controlled sweeps
------------------------------------------

Run calibration on the selected initialization, without a style module::

    python tools/stage3.py calibrate --manifest /data/calli/manifest.json \
      --data-root /data/calli --checkpoint /models/selected-stage2.pth \
      --output /runs/stage3-calibration

Uses 32 style-balanced batches of 2, fixed query masks and full precision.
Reports raw values, lower-query pixel gradients, trainable-parameter gradients,
pairwise cosine similarities and Gram matrices. Internal coefficients balance
median pixel gradients; a separate common scale preserves median combined
parameter-gradient norm. Coefficients outside [0.1, 100], zero gradients and
degenerate combined gradients block calibration. Do not increase limits to force
acceptance. Gradient-norm ratios are not independent update contribution shares.

The original structure definitions are unchanged: these compute global soft
foreground statistics of the concatenated image, while detail uses mask*valid.
The lower-query gradient measurement does not redefine the structure loss.

Generate commands for one phase at a time::

    python tools/stage3_experiments.py internal \
      --manifest /data/calli/manifest.json --data-root /data/calli \
      --checkpoint /models/selected-stage2.pth \
      --semantic-mask-dir /data/calli/font/train/new \
      --coefficients /runs/stage3-calibration/calibration.json \
      --output-root /runs/stage3 --output-json /runs/internal-commands.json

Each entry contains an argv list suitable for subprocess.run(argv, check=True).
It is not executed automatically. After visual/metric review, generate the next
phase with the selected weights and optional --coefficients:

* internal: equal coefficients vs calibrated, both weights 0.05.
* structure: total weight 0/0.025/0.05/0.1, detail fixed at 0.05.
* detail: total weight 0/0.025/0.05/0.1, selected structure unchanged.
* style: off/reference/constant, both selected losses unchanged.
* local: reference model, one selected loss at 0.5x or 2x at a time. A zero
  selected loss remains zero. Include original A/C when comparing final results.

Use --structure-weight and --detail-weight to carry explicit decisions. Omit
--coefficients if equal coefficients won. Use --seeds 1 2 to replicate finalists
and their baselines. Reuse a baseline only when checkpoint, data fingerprint,
all settings, seed and update budget match; command output directories never
overwrite previous runs.

run.json stores separate JSON-manifest and image/identity fingerprints. Semantic
mask contents are not included (semantic_masks_hashed=false); preserve their
dataset version separately and do not reuse a baseline after masks change.

Training defaults and compatibility
-----------------------------------

400 optimizer updates; effective batch 128 (2 GPUs * batch 2 * accumulation 32).
Single GPU requires --accum-iter 64. Freeze first 9 blocks and embeddings; no GAN.
LR 1e-4 with layer decay 0.8, new conditioner LR 3e-4, clipping 3.0. LR ramps for
40 updates then cosine decays. Structure/detail and edge ramp for 40 updates;
edge target is 0.3. Evaluate/save every 50 updates and at the final update.
The entrypoint passes mask random/JT/BF = 0.8/0/0.2 with half-mask probability
0.5 inside random mode, but PairDataset overrides the mode for every CalliPhase
record that has a semantic mask: JT records always sample one semantic layer and
BF records always sample eleven, so stage-3 training never uses random block or
half masks (decision 2026-09-21: keep this behaviour). The 0.8/0/0.2 setting
only affects records without semantic sources, which the manifest builder drops.
Training retains the existing weak finetune transform. Evaluation and both
existing inference entrypoints use square white padding and bicubic resize for
stage-3 checkpoints, without the old 64-pixel reference downsampling.

New stage-3 training defaults to --vgg-input-mode rgb: undo dataset normalization
before the VGG module's own normalization. --vgg-input-mode legacy is available
only for a controlled compatibility comparison; keep it identical across a
sweep. Existing training entrypoints retain their legacy default.

Style conditioning uses visible upper reference RGB plus visibility as four
channels; masked reference pixels are white and lower GT is never read. Three
Conv/GroupNorm/GELU stages (32/64/128), global pooling and three zero-initialized
heads modulate the last three ViT blocks. The constant branch receives identical
all-one inputs, with the same trainable architecture. No style-ID embedding is
used. Old checkpoints may miss only the new conditioner keys on migration;
other missing/unexpected keys and size mismatches are errors.

Reports and acceptance
----------------------

Each evaluation saves metrics.json and reference/prediction/GT strips. Metrics
use only lower query pixels: fixed Gaussian highpass, Sobel gradient error,
edge F1 (threshold 0.1, exact pixel matching), foreground geometry (threshold
0.9), centroid, row/column projection, area and bbox aspect. These are evaluation
metrics, not scaled copies of the training losses. Wrong/blank reference metrics
are diagnostic; aggregate only correct-reference cases::

    python tools/stage3.py summarize --metrics /runs/A/eval-400/metrics.json \
      /runs/B/eval-400/metrics.json --output /runs/comparison.json

Choose using fixed blind-review images and validation metrics together. Compare
matched seeds and train/validation gaps. Small changes below seed variability
are unconfirmed. Neither capacity, calibrated coefficients, detail weight nor
the new module is assumed effective. Stop expanding experiments if no candidate
improves the baseline. Register command, data hash, initialization, final selected
checkpoint, figures, metrics and human conclusion in AGENTS.md after real runs.

Local validation
----------------

    PYTHONPATH=. python -m pytest tests/test_stage3.py -q

Tests execute conditioning and actual encoder/loss method bodies with small
fixtures; the loss test substitutes the VGG extractor. They do not instantiate
the full pretrained ViT or establish CUDA/DDP training correctness.
