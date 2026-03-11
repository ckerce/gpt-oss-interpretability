# Filtered Benchmark Analysis

## Baseline Comparison

| Set | Count | Accuracy | Mean Margin |
| --- | ---: | ---: | ---: |
| all_cases | 20 | 0.650 | 2.883 |
| main_analysis_strict | 4 | 1.000 | 4.792 |
| main_analysis_soft | 9 | 1.000 | 6.260 |
| secondary_analysis_soft | 4 | 0.500 | 3.248 |
| failure_analysis | 7 | 0.286 | -1.667 |

## All Runs by Set

| Set | Run | Count | Accuracy | Mean Margin |
| --- | --- | ---: | ---: | ---: |
| all_cases | early_heads_L2@0 | 20 | 0.650 | 2.883 |
| all_cases | early_heads_L2@0.5 | 20 | 0.600 | 3.039 |
| all_cases | early_heads_L2@1 | 20 | 0.650 | 3.114 |
| all_cases | early_heads_L2@1.5 | 20 | 0.600 | 3.083 |
| all_cases | experts_L8@0 | 20 | 0.650 | 3.125 |
| all_cases | experts_L8@0.5 | 20 | 0.600 | 3.094 |
| all_cases | experts_L8@1 | 20 | 0.650 | 3.114 |
| all_cases | experts_L8@1.5 | 20 | 0.650 | 3.124 |
| all_cases | layer_scale_L20@0 | 20 | 0.500 | 0.359 |
| all_cases | layer_scale_L20@0.5 | 20 | 0.600 | 3.193 |
| all_cases | layer_scale_L20@1 | 20 | 0.650 | 3.114 |
| all_cases | layer_scale_L20@2 | 20 | 0.600 | 2.617 |
| all_cases | mid_heads_L12@0 | 20 | 0.650 | 3.178 |
| all_cases | mid_heads_L12@0.5 | 20 | 0.650 | 3.062 |
| all_cases | mid_heads_L12@1 | 20 | 0.650 | 3.114 |
| all_cases | mid_heads_L12@1.5 | 20 | 0.650 | 3.129 |
| main_analysis_strict | early_heads_L2@0 | 4 | 1.000 | 4.792 |
| main_analysis_strict | early_heads_L2@0.5 | 4 | 1.000 | 5.561 |
| main_analysis_strict | early_heads_L2@1 | 4 | 1.000 | 5.489 |
| main_analysis_strict | early_heads_L2@1.5 | 4 | 1.000 | 6.040 |
| main_analysis_strict | experts_L8@0 | 4 | 1.000 | 5.664 |
| main_analysis_strict | experts_L8@0.5 | 4 | 1.000 | 5.646 |
| main_analysis_strict | experts_L8@1 | 4 | 1.000 | 5.489 |
| main_analysis_strict | experts_L8@1.5 | 4 | 1.000 | 5.421 |
| main_analysis_strict | layer_scale_L20@0 | 4 | 0.500 | 0.831 |
| main_analysis_strict | layer_scale_L20@0.5 | 4 | 1.000 | 6.074 |
| main_analysis_strict | layer_scale_L20@1 | 4 | 1.000 | 5.489 |
| main_analysis_strict | layer_scale_L20@2 | 4 | 1.000 | 4.422 |
| main_analysis_strict | mid_heads_L12@0 | 4 | 1.000 | 5.664 |
| main_analysis_strict | mid_heads_L12@0.5 | 4 | 1.000 | 5.520 |
| main_analysis_strict | mid_heads_L12@1 | 4 | 1.000 | 5.489 |
| main_analysis_strict | mid_heads_L12@1.5 | 4 | 1.000 | 5.438 |
| main_analysis_soft | early_heads_L2@0 | 9 | 1.000 | 6.260 |
| main_analysis_soft | early_heads_L2@0.5 | 9 | 1.000 | 6.587 |
| main_analysis_soft | early_heads_L2@1 | 9 | 1.000 | 6.826 |
| main_analysis_soft | early_heads_L2@1.5 | 9 | 1.000 | 6.896 |
| main_analysis_soft | experts_L8@0 | 9 | 1.000 | 6.877 |
| main_analysis_soft | experts_L8@0.5 | 9 | 1.000 | 6.846 |
| main_analysis_soft | experts_L8@1 | 9 | 1.000 | 6.826 |
| main_analysis_soft | experts_L8@1.5 | 9 | 1.000 | 6.792 |
| main_analysis_soft | layer_scale_L20@0 | 9 | 0.444 | 0.945 |
| main_analysis_soft | layer_scale_L20@0.5 | 9 | 1.000 | 7.988 |
| main_analysis_soft | layer_scale_L20@1 | 9 | 1.000 | 6.826 |
| main_analysis_soft | layer_scale_L20@2 | 9 | 1.000 | 5.776 |
| main_analysis_soft | mid_heads_L12@0 | 9 | 1.000 | 7.039 |
| main_analysis_soft | mid_heads_L12@0.5 | 9 | 1.000 | 6.914 |
| main_analysis_soft | mid_heads_L12@1 | 9 | 1.000 | 6.826 |
| main_analysis_soft | mid_heads_L12@1.5 | 9 | 1.000 | 6.755 |
| secondary_analysis_soft | early_heads_L2@0 | 4 | 0.500 | 3.248 |
| secondary_analysis_soft | early_heads_L2@0.5 | 4 | 0.750 | 3.367 |
| secondary_analysis_soft | early_heads_L2@1 | 4 | 1.000 | 3.517 |
| secondary_analysis_soft | early_heads_L2@1.5 | 4 | 0.500 | 3.277 |
| secondary_analysis_soft | experts_L8@0 | 4 | 0.750 | 3.244 |
| secondary_analysis_soft | experts_L8@0.5 | 4 | 0.750 | 3.189 |
| secondary_analysis_soft | experts_L8@1 | 4 | 1.000 | 3.517 |
| secondary_analysis_soft | experts_L8@1.5 | 4 | 1.000 | 3.657 |
| secondary_analysis_soft | layer_scale_L20@0 | 4 | 0.750 | -3.322 |
| secondary_analysis_soft | layer_scale_L20@0.5 | 4 | 0.500 | 2.494 |
| secondary_analysis_soft | layer_scale_L20@1 | 4 | 1.000 | 3.517 |
| secondary_analysis_soft | layer_scale_L20@2 | 4 | 0.750 | 3.366 |
| secondary_analysis_soft | mid_heads_L12@0 | 4 | 1.000 | 3.370 |
| secondary_analysis_soft | mid_heads_L12@0.5 | 4 | 1.000 | 3.165 |
| secondary_analysis_soft | mid_heads_L12@1 | 4 | 1.000 | 3.517 |
| secondary_analysis_soft | mid_heads_L12@1.5 | 4 | 1.000 | 3.649 |
| failure_analysis | early_heads_L2@0 | 7 | 0.286 | -1.667 |
| failure_analysis | early_heads_L2@0.5 | 7 | 0.000 | -1.711 |
| failure_analysis | early_heads_L2@1 | 7 | 0.000 | -1.887 |
| failure_analysis | early_heads_L2@1.5 | 7 | 0.143 | -1.929 |
| failure_analysis | experts_L8@0 | 7 | 0.143 | -1.766 |
| failure_analysis | experts_L8@0.5 | 7 | 0.000 | -1.783 |
| failure_analysis | experts_L8@1 | 7 | 0.000 | -1.887 |
| failure_analysis | experts_L8@1.5 | 7 | 0.000 | -1.896 |
| failure_analysis | layer_scale_L20@0 | 7 | 0.429 | 1.710 |
| failure_analysis | layer_scale_L20@0.5 | 7 | 0.143 | -2.574 |
| failure_analysis | layer_scale_L20@1 | 7 | 0.000 | -1.887 |
| failure_analysis | layer_scale_L20@2 | 7 | 0.000 | -1.872 |
| failure_analysis | mid_heads_L12@0 | 7 | 0.000 | -1.896 |
| failure_analysis | mid_heads_L12@0.5 | 7 | 0.000 | -1.949 |
| failure_analysis | mid_heads_L12@1 | 7 | 0.000 | -1.887 |
| failure_analysis | mid_heads_L12@1.5 | 7 | 0.000 | -1.830 |
