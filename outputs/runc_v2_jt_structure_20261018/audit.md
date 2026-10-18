# Run C v2 data audit

Final correction: the initial path/hash split audit did not exclude query-self
training references. That defect invalidated all run1 training. It was fixed in
c54cf50 with `distinct-original-reference-v1`, exhaustive candidate-edge audits,
runtime identity guards, and clean restart from checkpoint14. See
data_reference_policy_audit.json, self_reference_replay_audit.json and
clean_run2/stage1_acceptance.md. Stage I uses the original 101 JT pairs only;
the unchanged historical BF139 contains 11 self pairs and is not used for this
stage-I assessment. The counts and native-coordinate audit below remain valid.

Status: **passed**. Recomputed from actual local files.

JT stage-I targets: 740 = 466 dual + 274 JT-only; BF-only=0. Raw identity / decoded pixel / target SHA unique counts: 740 / 740 / 740. Nine-author counts: {'Caoqb': 54, 'Dongqc': 68, 'Liugq': 96, 'Longmen': 51, 'Ouyx': 64, 'Sushi': 30, 'Yanzq': 231, 'Zhangqb': 51, 'Zhaomf': 95}.

Historical pairs: 240, JT=101; unique JT lower targets=77. Exact raw/pixel/target overlap=0/0/0. Same-author/character overlap=3: ['font/train/new/LiugqJT/images_text_denoised/賜.png', 'font/train/new/YanzqJT/images_text_denoised/海1.png', 'font/train/new/YanzqJT/images_text_denoised/歸2.png']. An additionally reported exclusion subset has 74 unique JT targets. It is not an independent generalization set.

Source evidence: `tools/build_dual_manifest.py:57` hashes the original file and `:58` forms the author/MD5 merge key; `data/dual_dataset.py:262` synchronizes crop across source/target and `:289` transforms instance layers; `:272` would sample BF masks even for dual targets. The new `data/runc_v2_dataset.py::RuncV2Dataset` builds stage-I pairs directly and never invokes that BF sampler. `get_pair` sets all 784 lower patches masked and zeroes the lower input-target pixels. GT images and instance arrays remain only as loss/evaluation fields. The new no-GT-leak data test substitutes the query GT and verifies image/input_target/mask invariance; model forward leakage requires its separate runtime check.

Native annotation mapping remains sealed `pad_square` in the existing cache. Head occupancy is derived at 28; raw J Voronoi at 224; coarse J Voronoi is independently derived at 112 from the same synchronously transformed instance ink, with partition loss flags preserved. These coordinate systems are separately reported.

Errors: []. Full per-target provenance is in data_manifest.json.
