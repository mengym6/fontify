# Run C v2 ia1 measured results

Export status: **complete_numerical_export**. Numerical export only; no winner selection or automatic structural acceptance.

Online loss uses sampled optimizer batches. Fixed train18, historical 101 JT pairs / 77 unique targets / 216 matched GT instances, and the same-author-character-excluded 74 unique targets are separate domains. Per-character means first average valid instances inside each word. Matched-instance tails pool eligible GT instances; relative-centroid counts can differ because two valid parts are required. Unique-target P90/P95 are quantiles across target means, not instance-tail quantiles.
For higher-is-better IoU/recall/accuracy, upper P90/P95 are not worst-case tails; for geometric errors upper quantiles indicate large-error tails.

The additional exclusion set removes the three target paths recorded in data_manifest; it remains reused validation and is not independent generalization. No slot-index alignment across references is invented.

Observed optimizer updates: 400; last step: 400.

## Snapshot evidence

| step | cohort | aggregation | pairs | unique | metric n | IoU mean | log sigma x mean | log sigma y mean | log sigma xy P95 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | historical_val | matched_instances | 101 | 77 | 216 | 0.252255 | 0.484584 | 0.416161 | 1.551508 |
| 0 | historical_val | unique_target_means | 101 | 77 | 77 | 0.253604 | 0.480915 | 0.413567 | 1.235373 |
| 0 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.251260 | 0.473746 | 0.424918 | 1.247811 |
| 0 | train18 | matched_instances | 18 | 18 | 40 | 0.229965 | 0.501254 | 0.416992 | 1.398668 |
| 0 | train18 | unique_target_means | 18 | 18 | 18 | 0.231402 | 0.494218 | 0.404803 | 1.256548 |
| 1 | historical_val | matched_instances | 101 | 77 | 216 | 0.243077 | 0.479332 | 0.408389 | 1.499957 |
| 1 | historical_val | unique_target_means | 101 | 77 | 77 | 0.244326 | 0.476958 | 0.405543 | 1.193211 |
| 1 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.242178 | 0.470070 | 0.415766 | 1.208340 |
| 1 | train18 | matched_instances | 18 | 18 | 40 | 0.222284 | 0.497875 | 0.408506 | 1.401584 |
| 1 | train18 | unique_target_means | 18 | 18 | 18 | 0.223411 | 0.490847 | 0.396461 | 1.265595 |
| 50 | historical_val | matched_instances | 101 | 77 | 216 | 0.315184 | 0.312390 | 0.320590 | 1.430707 |
| 50 | historical_val | unique_target_means | 101 | 77 | 77 | 0.317090 | 0.317660 | 0.308447 | 1.182577 |
| 50 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.311378 | 0.324781 | 0.312864 | 1.201195 |
| 50 | train18 | matched_instances | 18 | 18 | 40 | 0.322631 | 0.321659 | 0.285071 | 1.074743 |
| 50 | train18 | unique_target_means | 18 | 18 | 18 | 0.324180 | 0.318257 | 0.279539 | 0.941288 |
| 100 | historical_val | matched_instances | 101 | 77 | 216 | 0.386242 | 0.259693 | 0.284477 | 1.233976 |
| 100 | historical_val | unique_target_means | 101 | 77 | 77 | 0.387385 | 0.260632 | 0.279414 | 0.907574 |
| 100 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.382744 | 0.266476 | 0.281545 | 0.920960 |
| 100 | train18 | matched_instances | 18 | 18 | 40 | 0.382872 | 0.245998 | 0.236424 | 0.805291 |
| 100 | train18 | unique_target_means | 18 | 18 | 18 | 0.383196 | 0.244313 | 0.230055 | 0.829877 |
| 150 | historical_val | matched_instances | 101 | 77 | 216 | 0.396085 | 0.264004 | 0.270401 | 1.144131 |
| 150 | historical_val | unique_target_means | 101 | 77 | 77 | 0.397663 | 0.259397 | 0.272585 | 0.992639 |
| 150 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.393480 | 0.267379 | 0.274420 | 1.008026 |
| 150 | train18 | matched_instances | 18 | 18 | 40 | 0.404274 | 0.230525 | 0.226778 | 0.979186 |
| 150 | train18 | unique_target_means | 18 | 18 | 18 | 0.405142 | 0.227728 | 0.224092 | 0.716665 |
| 200 | historical_val | matched_instances | 101 | 77 | 216 | 0.445017 | 0.225100 | 0.255318 | 1.198828 |
| 200 | historical_val | unique_target_means | 101 | 77 | 77 | 0.454040 | 0.214991 | 0.244445 | 0.989663 |
| 200 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.448113 | 0.219937 | 0.249478 | 0.993353 |
| 200 | train18 | matched_instances | 18 | 18 | 40 | 0.448182 | 0.177198 | 0.236481 | 1.088816 |
| 200 | train18 | unique_target_means | 18 | 18 | 18 | 0.451357 | 0.171101 | 0.223749 | 0.897765 |
| 250 | historical_val | matched_instances | 101 | 77 | 216 | 0.439281 | 0.214185 | 0.228508 | 1.141777 |
| 250 | historical_val | unique_target_means | 101 | 77 | 77 | 0.445171 | 0.212277 | 0.222057 | 0.821934 |
| 250 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.440595 | 0.217658 | 0.224058 | 0.822781 |
| 250 | train18 | matched_instances | 18 | 18 | 40 | 0.464007 | 0.205642 | 0.165251 | 1.041504 |
| 250 | train18 | unique_target_means | 18 | 18 | 18 | 0.461469 | 0.195955 | 0.162833 | 0.731244 |
| 300 | historical_val | matched_instances | 101 | 77 | 216 | 0.446789 | 0.203972 | 0.229169 | 1.140438 |
| 300 | historical_val | unique_target_means | 101 | 77 | 77 | 0.454188 | 0.199055 | 0.221111 | 0.854897 |
| 300 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.449472 | 0.203162 | 0.223115 | 0.857472 |
| 300 | train18 | matched_instances | 18 | 18 | 40 | 0.464018 | 0.177334 | 0.185793 | 0.717862 |
| 300 | train18 | unique_target_means | 18 | 18 | 18 | 0.465465 | 0.173067 | 0.177542 | 0.571673 |
| 350 | historical_val | matched_instances | 101 | 77 | 216 | 0.454440 | 0.214957 | 0.232302 | 1.116137 |
| 350 | historical_val | unique_target_means | 101 | 77 | 77 | 0.461293 | 0.214496 | 0.225949 | 0.877761 |
| 350 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.456663 | 0.218921 | 0.227734 | 0.892621 |
| 350 | train18 | matched_instances | 18 | 18 | 40 | 0.473490 | 0.168833 | 0.179997 | 0.788462 |
| 350 | train18 | unique_target_means | 18 | 18 | 18 | 0.470597 | 0.164524 | 0.176679 | 0.556004 |
| 400 | historical_val | matched_instances | 101 | 77 | 216 | 0.449461 | 0.196248 | 0.230612 | 1.062357 |
| 400 | historical_val | unique_target_means | 101 | 77 | 77 | 0.455331 | 0.199919 | 0.220414 | 0.732686 |
| 400 | same_author_character_excluded_val | unique_target_means | 98 | 74 | 74 | 0.451825 | 0.202965 | 0.221166 | 0.738810 |
| 400 | train18 | matched_instances | 18 | 18 | 40 | 0.473567 | 0.193528 | 0.167676 | 1.013225 |
| 400 | train18 | unique_target_means | 18 | 18 | 18 | 0.469626 | 0.183972 | 0.166220 | 0.693676 |

## Coordinates and losses

Head geometry uses soft occupancy on 28x28 with the original Hungarian assignment. Density/occupancy and training J use 112x112 and the directly built coarse GT partition. P0/oracle-original448 metrics share the original448 GT coordinate system and a common GT Voronoi expanded from224. Oracle-original448 is representation error; it is not a generated model output. BCE/Dice and weighted occupancy objective remain separately labeled in CSV. Raw inactive coarse zeros in IA are not presented as rendered image measurements.

## Provenance

```json
{
  "training_source_sha": "c54cf5086a5fa8e4467da489f4bab3e27838e165",
  "start_checkpoint_sha256": "5b380960bdc6c1e8046178bab807998b2252d8566e7e1bd9e20886e307368dfb",
  "initial_checkpoint_sha256": "339ae9781f4f3831b9c97a19f3c5160c72fa790dbc82157d1d7d6ddf7be17dac",
  "evaluation_source_shas": [
    "c54cf5086a5fa8e4467da489f4bab3e27838e165"
  ],
  "snapshot_file_shas": {
    "0": "d058e852ee216ebc94f08e09076235343b721a4204d16e2a59b1a5d98bb24765",
    "1": "fd468318fd5fff5c80249b8240e4e76cb510eebffd3b39f8c9d0d2ce80b3d5a2",
    "50": "e55b94625117660e3c483d0ac2e863cc29c9c2e27dff8849dbfba69b2d8debee",
    "100": "fcc0070f4c3e4472128abc1c17c6da0a13ab9dc9403af17d4f3f657869d6faed",
    "150": "95d404aa42b8208b72b4854168935a8b4ea7e24fe2ae1c9150d1ba2c5b6cf3be",
    "200": "9b8a03efe3d511c20d162d35951d3d9a852be04a923464a533411f8052902d1a",
    "250": "2dbb76c0d08dc67e5a23a4d32210b9395d3c716269ee6486e68c38717c6eab80",
    "300": "4bb25049c7a9a2d2e158c8c821148d46175e8e44668c8629cb09a3008b40b223",
    "350": "1c6065d2f2cd87419b24a4783ff28d2df32ffa23d61e28d5b1de7750cc562cb4",
    "400": "c2da2b87323cedfc3853acb414eb1dc3553a56bf67f8fab5e58d425fd2b7e757"
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
      "step": 50,
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
      "step": 50,
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
      "step": 100,
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
      "step": 100,
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
      "step": 150,
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
      "step": 150,
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
      "step": 250,
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
      "step": 250,
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
      "step": 300,
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
      "step": 300,
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
      "step": 350,
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
      "step": 350,
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
      "step": 400,
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
      "step": 400,
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
