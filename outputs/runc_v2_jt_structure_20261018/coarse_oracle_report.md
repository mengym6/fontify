# Run C v2 density / occupancy oracle

Zero training. All inputs below are supervision-side targets, never deployment model conditions.

RGB is de-normalized once; grayscale is .299R+.587G+.114B. Density is avg_pool4(1-gray); occupancy is max_pool4(ink>0.5), aligning with legacy 8-bit gray<128 at grayscale values 127/128. Density retains antialias ink and background gray continuously.

Threshold audit uses two deterministic JT-bearing train targets per author (18 total), with ink fraction/component/noise statistics at tau .25/.5/.75. Legacy-vs-tau.5 pixel disagreements: 0. See data_manifest.json threshold_train_probe.

77 unique historical JT targets evaluated; twelve fixed visual targets cover nine authors including 無/容/惟/雲/崇. Raw224 geometry means: {'J_raw224_equal_coefficients': 0.0072501706467433405, 'logsigma_x': 0.0010586295831742903, 'logsigma_y': 0.0010769911644853312, 'shape': 0.005041940264742483, 'centroid_relative': 7.259051002998226e-05, 'centroid_absolute_normalized': 1.9439216815682173e-05}. J here uses centroid/logsigma/shape coefficients 1/1/1, a constant GT Voronoi at 224, and per-unique-target averaging; it must not be numerically compared to historical weighted Run B or v1 J. No prediction is thresholded for this geometric calculation.

Connected-component sensitivity: {'0.25': {'tiny_le64_count': 50, 'tiny_le64_lost': 37, 'tiny_retention': 0.26, 'all_components_lost': 37, 'merged_output_components': 21}, '0.5': {'tiny_le64_count': 33, 'tiny_le64_lost': 33, 'tiny_retention': 0.0, 'all_components_lost': 33, 'merged_output_components': 4}, '0.75': {'tiny_le64_count': 38, 'tiny_le64_lost': 38, 'tiny_retention': 0.0, 'all_components_lost': 38, 'merged_output_components': 0}}. Components use 4-neighbor connectivity. A source component disappears when no oracle pixel above the same threshold intersects it; split/merge counts use intersecting source/output labels. Small<=256 and tiny<=64 source pixels are descriptive size strata, not acceptance thresholds. Tiny fragments can be annotation/denoising noise rather than meaningful strokes; the full CSV and images are required for interpretation.

At tau .5 the 33 disappearing components have at most 15 source pixels each; their exact bboxes and oracle peaks are in data_oracle_lost_components.json. The fixed Caoqb 無 has five GT and five oracle components (main structure plus four bottom dots), with zero loss/merge at all three thresholds. At tau .5 narrow-gap merges occur in Dongqc 惟, Sushi 容, Yanzq 海, and Zhangqb 新. Their presence is a representation limitation even when mean J is small.

The density representation preserves mass (avg pooling exactly up to floating-point error), yet thresholded topology can change because bilinear interpolation lowers isolated point peaks and joins narrow gaps. Occupancy is auxiliary; it is not written into P0. Thus its retention must not be mistaken for density-P0 retention.

Review gate: these numerical diagnostics do not substitute for human judgment about bottom dots and short strokes. Inspect visuals/oracle/index.html before accepting 112 as adequate. The oracle is an attainable representation floor; trained model quality has not been established.
