# Run C v2 ib measured results

Export status: **complete_numerical_export**. Numerical export only; no winner selection or automatic structural acceptance.

Online loss uses sampled optimizer batches. Fixed train18, historical 101 JT pairs / 77 unique targets / 216 matched GT instances, and the same-author-character-excluded 74 unique targets are separate domains. Per-character means first average valid instances inside each word. Matched-instance tails pool eligible GT instances; relative-centroid counts can differ because two valid parts are required. Unique-target P90/P95 are quantiles across target means, not instance-tail quantiles.
For higher-is-better IoU/recall/accuracy, upper P90/P95 are not worst-case tails; for geometric errors upper quantiles indicate large-error tails.

The additional exclusion set removes the three target paths recorded in data_manifest; it remains reused validation and is not independent generalization. No slot-index alignment across references is invented.

Observed optimizer updates: 320; last step: 320.

## Snapshot evidence

| step | cohort | aggregation | pairs | unique | metric n | IoU mean | log sigma x mean | log sigma y mean | log sigma xy P95 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 0 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 0 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 0 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 0 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 1 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 1 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 1 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 1 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 1 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 40 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 40 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 40 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 40 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 40 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 80 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 80 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 80 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 80 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 80 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 120 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 120 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 120 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 120 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 120 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 160 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 160 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 160 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 160 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 160 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 200 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 200 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 200 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 200 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 200 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 240 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 240 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 240 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 240 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 240 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 280 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 280 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 280 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 280 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 280 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |
| 320 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 320 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 320 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 320 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 320 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |

## Coordinates and losses

Head geometry uses soft occupancy on 28x28 with the original Hungarian assignment. Density/occupancy and training J use 112x112 and the directly built coarse GT partition. P0/oracle-original448 metrics share the original448 GT coordinate system and a common GT Voronoi expanded from224. Oracle-original448 is representation error; it is not a generated model output. BCE/Dice and weighted occupancy objective remain separately labeled in CSV. Raw inactive coarse zeros in IA are not presented as rendered image measurements.

## Provenance

```json
{
  "training_source_sha": "6411b2675834e5a2aa623635161c2a4eb6458123",
  "start_checkpoint_sha256": "c2da2b87323cedfc3853acb414eb1dc3553a56bf67f8fab5e58d425fd2b7e757",
  "initial_checkpoint_sha256": "339ae9781f4f3831b9c97a19f3c5160c72fa790dbc82157d1d7d6ddf7be17dac",
  "evaluation_source_shas": [
    "6411b2675834e5a2aa623635161c2a4eb6458123"
  ],
  "snapshot_file_shas": {
    "0": "9725a1a86109928986f0526f4911e93375ccb26570efff0eee0ed693c113334a",
    "1": "43ce5a2f01fd17ee4295d9d2f724c57510fcae34f2d91b75962675c6906290c8",
    "40": "0556133c954afcd00f547872be26eab037067fcba973aa3a6006c13771b2c5ae",
    "80": "c1ed140f0ef64ff2a0e0fe39ebe7d9606e009b3c244acf5e2a576a3beaa9dba2",
    "120": "ba9038775aa0a1932994249772624dffa806ed4ad3ed52a8fdff77c82f639445",
    "160": "5fae2e9e53d3745fbac12522c1223f402e56238d5d1768e6cb39085af8e2090d",
    "200": "704ab55696b6de5e4849a6e6cb47bb77d245f0c8a5a0493807c4413733a4303b",
    "240": "84a81510f5bd8b7e94070fd1c9d501005ee22be8cbe8ec7122b8df6d5531e7ce",
    "280": "f8b8e7732a8ba178b4afcc916863078be3341e950bf249831d320202095e8cbc",
    "320": "f3e39d8f1747ae5449c74e2a83f5d189d00b49edbc3b4da54f25d87b991970d4"
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
      "step": 40,
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
      "step": 40,
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
      "step": 80,
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
      "step": 80,
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
      "step": 120,
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
      "step": 120,
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
      "step": 160,
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
      "step": 160,
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
      "step": 200,
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
      "step": 200,
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
      "step": 240,
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
      "step": 240,
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
      "step": 280,
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
      "step": 280,
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
      "step": 320,
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
      "step": 320,
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
