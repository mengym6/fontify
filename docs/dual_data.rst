Run B data preparation and instance targets
==========================================

Train inputs are the versioned ``train_json_clean_v2`` lists. The manifest
builder merges only train by calligrapher and original ``images/`` file MD5;
double-annotated records use the BF denoised image as their sole image GT.
The local manifest receipt is 1092 records: 466 dual, 352 BF only, 274 JT only.
Server images must be checked and the manifest rebuilt there; these local
counts and image hashes do not establish the server data version.

Commands (output directories must not already exist)::

    python tools/build_dual_manifest.py \
      --data_path fontdata_example \
      --input_dir fontdata_example/train_json_clean_v2 \
      --output_dir fontdata_example/train_json_dual_v1

    python tools/build_dual_targets.py \
      --data_path fontdata_example \
      --json_paths fontdata_example/train_json_dual_v1/train_dual.json \
                   fontdata_example/val_json_clean_v2 \
      --fixed_pair_path fontdata_example/val_pairs_fixed_clean.json \
      --output_dir fontdata_example/dual_targets_v1

Caches contain original 448-square per-instance regions, JT text, and classes;
they do not use existing JT semantic NPY layers. Instances are rendered from
COCO and deduplicated at region IoU strictly greater than 0.9 before text
intersection. Every image, BF mask, region, and text layer uses the same crop.
Intersection, validity, 28-grid occupancy and 224-grid J partition are then
derived from those transformed layers. Fixed validation preserves its original
pair records and uses deterministic identity geometry, without merging.

Fresh caches use ``schema=dual-target-cache-v2`` and
``source_seal=all-record-inputs-v1``. ``source_inputs`` records each train,
validation, and fixed-reference row's role-to-path mapping; ``source_sha256``
hashes each distinct path once. This covers computer/source images, denoised
image GTs and originals for BF-only as well as JT records, all BF semantic NPY
inputs, and JT COCO/raw/GT paths. The cache builder reads fixed-reference values
itself, even when those values are absent from the query JSON inputs. BF COCO
annotations are not consumed by the JT renderer and never supply JT text.

With ``verify_sources=True``, dataset construction verifies every relevant query
and reference input against those hashes. Missing seals, missing input hashes,
or modified source files fail. The old ``dual-target-cache-v1`` format remains
readable for local compatibility but reports
``source_verification_scope=jt-cache-only-legacy``: its checks cover only cached
JT COCO/raw/GT, so it cannot establish complete training-data identity. Rebuild
the server cache with v2 for a full source seal. Disabling source verification
reports ``source_verification_scope=disabled``. Fixed validation additionally
requires each fixed query key to occur exactly once in the query JSON inputs.

Occupancy retains independent overlapping instance ink. For J only, overlapping
ink receives unique ownership using the existing ``ink_label_map`` rule: later
instance layers cover earlier layers. The ownership label map is downsampled
by nearest neighbor to 224, then exact squared EDT builds the Voronoi partition.
This keeps the existing J overlap/nearest-grid contract. Deduplication order is
stable ascending COCO annotation ID and is recorded in the cache index.

The dataset separately reports ``j_downsample_lost`` (a 448-valid instance has
no independent ink on the 224 nearest grid) and ``j_partition_lost`` (nearest
ink exists but overlap ownership or EDT gives it no partition). These diagnostic
fields do not change H validity: H still sees 28-grid independent occupancy,
and J retains its existing GT-mass shielding. ``downsample_audit.json`` records
both conditions for unaugmented cached targets. Runtime fields additionally
cover augmentation-dependent losses.

The local 869-target, 1988-valid-instance audit has zero losses in either
category with this ownership rule. An earlier independent-EDT implementation
lost the inner instance of Yanzq ``圎`` because its ink was contained in the
outer instance and every distance tie favored the outer instance. Its inner
instance has 22880 pixels at 448 and 5711 on the nearest 224 grid; that loss was
an overlap ownership problem, not a downsampling problem. The existing label-map
rule retains both inner and outer J regions. No raw 448 cache rebuild is needed
when switching to the documented ownership derivation.

``DualDataset.get_pair(index, ref_index=None, augmentation_seed=None)`` supports
fixed probes without advancing the Python, NumPy, or torch CPU RNG. Train index
denotes the lower query; the upper reference is uniform in the same author/pool.
Validation retains the previous fixed-file row order. Scheme D weights are
``1 / sqrt(pool_size)`` per record, so each author/pool gets mass ``sqrt(n)``.
