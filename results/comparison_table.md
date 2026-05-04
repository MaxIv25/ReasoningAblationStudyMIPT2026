
### Accuracy (%)

| Experiment | gsm8k | math500 | math_hard |
| --- | --- | --- | --- |
| base_model | 55.42 | 38.6 | 17.98 |
| exp2_1_full_ft | 64.82 | 45.6 | 22.36 |
| exp2_2_lora | 64.82 | 45.8 | 19.94 |
| exp2_3_dora | 64.82 | 45.8 | 21.45 |
| exp2_4_pissa | 64.82 | 42.2 | 21.6 |
| exp3_1_easy2hard | 64.82 | 42.6 | 22.43 |
| exp3_2_random | 64.52 | 45.8 | 19.71 |
| exp3_3_hard2easy | 64.82 | 44.8 | 19.86 |

### Format Compliance (%)

| Experiment | gsm8k | math500 | math_hard |
| --- | --- | --- | --- |
| base_model | 91.66 | 74.2 | 63.52 |
| exp2_1_full_ft | 93.56 | 75.2 | 57.33 |
| exp2_2_lora | 93.56 | 72.8 | 54.83 |
| exp2_3_dora | 93.56 | 73.4 | 56.65 |
| exp2_4_pissa | 93.56 | 73.2 | 55.44 |
| exp3_1_easy2hard | 93.56 | 70.0 | 56.19 |
| exp3_2_random | 94.69 | 74.0 | 56.19 |
| exp3_3_hard2easy | 93.56 | 75.8 | 55.59 |

### Average Response Length (words)

| Experiment | gsm8k | math500 | math_hard |
| --- | --- | --- | --- |
| base_model | 665.4 | 2703.3 | 3714.4 |
| exp2_1_full_ft | 1490.3 | 3065.3 | 4778.3 |
| exp2_2_lora | 1490.3 | 3134.9 | 4974.1 |
| exp2_3_dora | 1490.3 | 3102.3 | 4829.5 |
| exp2_4_pissa | 1490.3 | 3182.0 | 4881.9 |
| exp3_1_easy2hard | 1490.3 | 3380.1 | 4852.4 |
| exp3_2_random | 1401.5 | 3301.7 | 4945.1 |
| exp3_3_hard2easy | 1490.3 | 2985.2 | 4896.6 |
