# Run C v2 ia0 measured results

Export status: **complete_numerical_export**. Numerical export only; no winner selection or automatic structural acceptance.

Online loss uses sampled optimizer batches. Fixed train18, historical 101 JT pairs / 77 unique targets / 216 matched GT instances, and the same-author-character-excluded 74 unique targets are separate domains. Per-character means first average valid instances inside each word. Matched-instance tails pool eligible GT instances; relative-centroid counts can differ because two valid parts are required. Unique-target P90/P95 are quantiles across target means, not instance-tail quantiles.
For higher-is-better IoU/recall/accuracy, upper P90/P95 are not worst-case tails; for geometric errors upper quantiles indicate large-error tails.

The additional exclusion set removes the three target paths recorded in data_manifest; it remains reused validation and is not independent generalization. No slot-index alignment across references is invented.

Observed optimizer updates: 4; last step: 4.

## Snapshot evidence

| step | cohort | aggregation | pairs | unique | metric n | IoU mean | log sigma x mean | log sigma y mean | log sigma xy P95 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | historical_val | matched_instances | 101 | 77 | 216 | 0.131635 | 0.837876 | 0.743882 | 2.430211 |
| 0 | historical_val | unique_target_means | 101 | 77 | 77 | 0.133017 | 0.828429 | 0.745469 | 2.060794 |
| 0 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.130716 | 0.821256 | 0.764626 | 2.068771 |
| 0 | train18 | matched_instances | 18 | 18 | 40 | 0.120334 | 0.925396 | 0.763702 | 2.323879 |
| 0 | train18 | unique_target_means | 18 | 18 | 18 | 0.122138 | 0.912609 | 0.756982 | 1.979365 |
| 1 | historical_val | matched_instances | 101 | 77 | 216 | 0.140647 | 0.583352 | 0.546561 | 1.907195 |
| 1 | historical_val | unique_target_means | 101 | 77 | 77 | 0.140795 | 0.570089 | 0.546686 | 1.635296 |
| 1 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.139470 | 0.561551 | 0.567037 | 1.635925 |
| 1 | train18 | matched_instances | 18 | 18 | 40 | 0.127348 | 0.633951 | 0.567303 | 1.801746 |
| 1 | train18 | unique_target_means | 18 | 18 | 18 | 0.127595 | 0.625213 | 0.556783 | 1.456148 |
| 4 | historical_val | matched_instances | 101 | 77 | 216 | 0.252255 | 0.484584 | 0.416161 | 1.551508 |
| 4 | historical_val | unique_target_means | 101 | 77 | 77 | 0.253604 | 0.480915 | 0.413567 | 1.235373 |
| 4 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.251260 | 0.473746 | 0.424918 | 1.247811 |
| 4 | train18 | matched_instances | 18 | 18 | 40 | 0.229965 | 0.501254 | 0.416992 | 1.398668 |
| 4 | train18 | unique_target_means | 18 | 18 | 18 | 0.231402 | 0.494218 | 0.404803 | 1.256548 |

## Coordinates and losses

Head geometry uses soft occupancy on 28x28 with the original Hungarian assignment. Density/occupancy and training J use 112x112 and the directly built coarse GT partition. P0/oracle-original448 metrics share the original448 GT coordinate system and a common GT Voronoi expanded from224. Oracle-original448 is representation error; it is not a generated model output. BCE/Dice and weighted occupancy objective remain separately labeled in CSV. Raw inactive coarse zeros in IA are not presented as rendered image measurements.

## Provenance

```json
{
  "training_source_sha": "c54cf5086a5fa8e4467da489f4bab3e27838e165",
  "start_checkpoint_sha256": "339ae9781f4f3831b9c97a19f3c5160c72fa790dbc82157d1d7d6ddf7be17dac",
  "initial_checkpoint_sha256": "339ae9781f4f3831b9c97a19f3c5160c72fa790dbc82157d1d7d6ddf7be17dac",
  "evaluation_source_shas": [
    "c54cf5086a5fa8e4467da489f4bab3e27838e165"
  ],
  "snapshot_file_shas": {
    "0": "b50a8527dbaf153c7a6208155b11abc4270150dc895df67e1b0ae76e113c0272",
    "1": "d0d8885ea8139f6594ab97f9ee389590a0fb83524f3bf90472815683874d8241",
    "4": "5b380960bdc6c1e8046178bab807998b2252d8566e7e1bd9e20886e307368dfb"
  },
  "data_manifest_sha256": "cb8e11c8ee87daf67f78529bad682256bf6460c1f7ceba613ffda6a2cd430848",
  "training_backend": {
    "torch": "2.3.0+cu121",
    "cuda": "12.1",
    "cudnn": 8902,
    "device": "cuda:0",
    "gpu": "NVIDIA GeForce RTX 4090",
    "matmul_tf32": false,
    "cudnn_tf32": false,
    "cudnn_benchmark": false,
    "deterministic_algorithms": false
  },
  "evaluation_backends": [
    {
      "step": 0,
      "cohort": "historical_val",
      "backend": {
        "torch": "2.3.0+cu121",
        "cuda": "12.1",
        "cudnn": 8902,
        "device": "cuda:0",
        "gpu": "NVIDIA GeForce RTX 4090",
        "matmul_tf32": false,
        "cudnn_tf32": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false
      },
      "precision": "native FP32 forward; model eval; TF32 per backend record"
    },
    {
      "step": 0,
      "cohort": "train18",
      "backend": {
        "torch": "2.3.0+cu121",
        "cuda": "12.1",
        "cudnn": 8902,
        "device": "cuda:0",
        "gpu": "NVIDIA GeForce RTX 4090",
        "matmul_tf32": false,
        "cudnn_tf32": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false
      },
      "precision": "native FP32 forward; model eval; TF32 per backend record"
    },
    {
      "step": 1,
      "cohort": "historical_val",
      "backend": {
        "torch": "2.3.0+cu121",
        "cuda": "12.1",
        "cudnn": 8902,
        "device": "cuda:0",
        "gpu": "NVIDIA GeForce RTX 4090",
        "matmul_tf32": false,
        "cudnn_tf32": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false
      },
      "precision": "native FP32 forward; model eval; TF32 per backend record"
    },
    {
      "step": 1,
      "cohort": "train18",
      "backend": {
        "torch": "2.3.0+cu121",
        "cuda": "12.1",
        "cudnn": 8902,
        "device": "cuda:0",
        "gpu": "NVIDIA GeForce RTX 4090",
        "matmul_tf32": false,
        "cudnn_tf32": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false
      },
      "precision": "native FP32 forward; model eval; TF32 per backend record"
    },
    {
      "step": 4,
      "cohort": "historical_val",
      "backend": {
        "torch": "2.3.0+cu121",
        "cuda": "12.1",
        "cudnn": 8902,
        "device": "cuda:0",
        "gpu": "NVIDIA GeForce RTX 4090",
        "matmul_tf32": false,
        "cudnn_tf32": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false
      },
      "precision": "native FP32 forward; model eval; TF32 per backend record"
    },
    {
      "step": 4,
      "cohort": "train18",
      "backend": {
        "torch": "2.3.0+cu121",
        "cuda": "12.1",
        "cudnn": 8902,
        "device": "cuda:0",
        "gpu": "NVIDIA GeForce RTX 4090",
        "matmul_tf32": false,
        "cudnn_tf32": false,
        "cudnn_benchmark": false,
        "deterministic_algorithms": false
      },
      "precision": "native FP32 forward; model eval; TF32 per backend record"
    }
  ]
}
```

Evaluation metadata checkpoint SHA identifies the starting checkpoint. Snapshot file SHA comes from the saved sidecar manifest; evaluated parameter-state SHA identifies the model actually evaluated. Model binaries were not read by this exporter.

## Limitations and missing evidence

- No missing numerical artifacts detected for the requested budget.
- Single-seed observations; repeated historical validation selection.
- Human review and final structure acceptance are not performed here.
