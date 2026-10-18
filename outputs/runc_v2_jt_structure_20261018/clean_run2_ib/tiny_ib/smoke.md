# Run C v2 fixed real-JT tiny health

Stage=ib, status=passed, optimizer steps=12. Two targets/authors: ['Caoqb', 'Dongqc'] / ['述', '待']. Reference and augmentation were sampled once and the same tensor batch was reused throughout. Each microbatch contributes its actual target count divided by two. No checkpoint is saved or reused.

Before objective/H/Hgeo: 3.746263 / 0.886255 / 1.137222; after: 1.649603 / 0.886255 / 1.137222. Decreases: {'total': True, 'raw_head': False, 'raw_head_geo': False}.

Frozen parameters and buffers unchanged=True; expected active groups changed=True. Exact group/tensor hashes are in freeze_sha.json; gradient norms and loss axes are in smoke.json.

Full S/D0/O0/P0 forward shapes, parameter counts, measured CUDA peak memory and profiler-counted operator FLOPs are recorded separately. Fused SDPA and unsupported operators may be uncounted; this is not a complete MACs/FLOPs claim. The health run provides no quality or independent-generalization conclusion.
