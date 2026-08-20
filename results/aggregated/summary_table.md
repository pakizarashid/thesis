# Aggregated results summary

Generated from 50 files in results/. Values under 'this project' are measured; the two reference blocks below are literal transcriptions of published numbers, not measured here.


## Audio quality (PESQ/STOI/SI-SNR)

| label                                         | sample_dir                                    |   mean_pesq |   mean_stoi |   mean_snr_db |   mean_wer | reference_text                             |   mean_si_snr_db |
|:----------------------------------------------|:----------------------------------------------|------------:|------------:|--------------:|-----------:|:-------------------------------------------|-----------------:|
| ./audio_samples/baseline                      | ./audio_samples/baseline                      |      2.0427 |      0.902  |        3.9628 |     0      | This is a test sentence for voice cloning. |         nan      |
| ./audio_samples/baseline_large                | ./audio_samples/baseline_large                |      2.15   |      0.9133 |        3.806  |     0.035  | This is a test sentence for voice cloning. |         nan      |
| ./audio_samples/baseline_n50                  | ./audio_samples/baseline_n50                  |      2.1971 |      0.9103 |        3.4258 |     0.045  | This is a test sentence for voice cloning. |           3.2565 |
| ./audio_samples/low_perturbation_n50          | ./audio_samples/low_perturbation_n50          |      2.3827 |      0.9165 |        4.6574 |     0.04   | This is a test sentence for voice cloning. |           3.254  |
| ./audio_samples/stage1_low_perturbation       | ./audio_samples/stage1_low_perturbation       |      2.2068 |      0.9061 |        4.9933 |     0.0625 | This is a test sentence for voice cloning. |         nan      |
| ./audio_samples/stage1_low_perturbation_large | ./audio_samples/stage1_low_perturbation_large |      2.3814 |      0.9188 |        5.0799 |     0.04   | This is a test sentence for voice cloning. |         nan      |


## AudioPure purification

| label                                                                  |   acc_before |   acc_after |   acc_drop |
|:-----------------------------------------------------------------------|-------------:|------------:|-----------:|
| baseline                                                               |       0.9825 |      0.4875 |     0.495  |
| ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt        |       1      |      0.4925 |     0.5075 |
| ./checkpoints/stage1_aug/stage1_epoch29.pt                             |       0.9875 |      0.4925 |     0.495  |
| ./checkpoints/stage1_full/stage1_epoch29.pt                            |       1      |      0.4825 |     0.5175 |
| ./checkpoints/stage2_sim_longrun/stage2_epoch29.pt                     |       0.995  |      0.495  |     0.5    |
| ./checkpoints/stage1_low_perturbation/low_perturbation_final.pt        |       1      |      0.5325 |     0.4675 |
| ./checkpoints/stage2_sim_longrun_recalibrated/recalibrated_final.pt    |       0.995  |      0.4825 |     0.5125 |
| ./checkpoints/stage2_sim_longrun_recalibrated_v2/recalibrated_final.pt |       0.995  |      0.4925 |     0.5025 |


## Augmentation robustness

| label                                                           |    clean |   masking |   shuffling |   replacing |   neural |   vctk_detection_acc |
|:----------------------------------------------------------------|---------:|----------:|------------:|------------:|---------:|---------------------:|
| ./checkpoints/stage1_aug/stage1_epoch29.pt                      |   0.9777 |    0.9911 |      0.9821 |      0.9732 |   0.8638 |             nan      |
| baseline                                                        |   0.9866 |    0.9888 |      0.9866 |      0.9621 |   0.9509 |             nan      |
| ./checkpoints/stage1_full/stage1_epoch29.pt                     |   0.9799 |    0.9799 |      0.9777 |      0.9509 |   0.8884 |             nan      |
| baseline                                                        | nan      |  nan      |    nan      |    nan      | nan      |               0.9978 |
| ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt | nan      |  nan      |    nan      |    nan      | nan      |               0.9978 |


## Disruption (SIM)

| label                                                                                      | sample_dir                         |   mean_pesq |   mean_stoi |   mean_snr_db |   mean_wer | reference_text                             |      sim |   pivotal_distance |   sim_before |   sim_after |   sim_drop |   pivotal_before |   pivotal_after |   detection_acc_before |   detection_acc_after |   detection_acc_drop |   perturbation_linf_mean |   perturbation_snr_db_mean |
|:-------------------------------------------------------------------------------------------|:-----------------------------------|------------:|------------:|--------------:|-----------:|:-------------------------------------------|---------:|-------------------:|-------------:|------------:|-----------:|-----------------:|----------------:|-----------------------:|----------------------:|---------------------:|-------------------------:|---------------------------:|
| baseline                                                                                   | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4246 |             2.2107 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage2_sim_longrun_recalibrated_v2/recalibrated_final.pt                     | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4448 |             2.1913 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| baseline                                                                                   | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4704 |             1.8514 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| baseline                                                                                   | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4552 |             1.8747 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| baseline                                                                                   | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.451  |             1.8734 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| pgd_eps0.001_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.2884 |           nan      |       0.4661 |      0.2884 |     0.1777 |           1.8657 |          1.7939 |                 0.9952 |                1      |              -0.0048 |                    0.001 |                    39.323  |
| pgd_eps0.002_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.2234 |           nan      |       0.4544 |      0.2234 |     0.2309 |           2.0047 |          2.0241 |                 0.9969 |                0.9988 |              -0.0019 |                    0.002 |                    32.6713 |
| pgd_eps0.002_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.2507 |           nan      |       0.4539 |      0.2507 |     0.2032 |           1.8644 |          1.7963 |                 0.9952 |                0.9952 |               0      |                    0.002 |                    33.3035 |
| pgd_eps0.003_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.2308 |           nan      |       0.4485 |      0.2308 |     0.2176 |           1.8443 |          1.8033 |                 0.9952 |                0.9928 |               0.0024 |                    0.003 |                    29.7926 |
| pgd_eps0.005_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.2204 |           nan      |       0.461  |      0.2204 |     0.2405 |           1.8577 |          1.7551 |                 0.9952 |                0.9928 |               0.0024 |                    0.005 |                    25.3937 |
| pgd_eps0.007_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.2199 |           nan      |       0.4567 |      0.2199 |     0.2368 |           1.8715 |          1.9096 |                 0.9952 |                0.9904 |               0.0048 |                    0.007 |                    22.4703 |
| pgd_eps0.01_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt  | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.1984 |           nan      |       0.4569 |      0.1984 |     0.2584 |           2.0217 |          2.0346 |                 0.9969 |                0.9875 |               0.0094 |                    0.01  |                    18.7501 |
| pgd_eps0.01_n10_lwm1.0_on_./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt  | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.2214 |           nan      |       0.4675 |      0.2214 |     0.2461 |           1.8841 |          1.8471 |                 0.9952 |                0.9928 |               0.0024 |                    0.01  |                    19.3834 |
| ./checkpoints/stage1_full_recalibrated/recalibrated_final.pt                               | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4881 |             1.8611 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage1_full_recalibrated_v2/recalibrated_final.pt                            | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4737 |             1.8798 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage1_full/stage1_epoch29.pt                                                | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4593 |             1.8662 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage1_full/stage1_epoch29.pt                                                | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4664 |             1.8688 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage1_full/stage1_epoch29.pt                                                | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4695 |             1.8949 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage2_sim_longrun/stage2_epoch29.pt                                         | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4654 |             1.8536 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage2_sim_longrun/stage2_epoch29.pt                                         | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4803 |             1.8498 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage2_sim_longrun/stage2_epoch29.pt                                         | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4735 |             1.8574 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./checkpoints/stage2_sim_longrun_recalibrated/recalibrated_final.pt                        | nan                                |      nan    |    nan      |      nan      |   nan      | nan                                        |   0.4671 |             1.8686 |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |
| ./audio_samples/stage2_sim_longrun                                                         | ./audio_samples/stage2_sim_longrun |        2.18 |      0.9042 |        4.9038 |     0.0312 | This is a test sentence for voice cloning. | nan      |           nan      |     nan      |    nan      |   nan      |         nan      |        nan      |               nan      |              nan      |             nan      |                  nan     |                   nan      |


## False positive rate

| label                                                                  |   false_positive_rate |   mean_presence_logit |
|:-----------------------------------------------------------------------|----------------------:|----------------------:|
| baseline                                                               |                  0.04 |               -7.2496 |
| ./checkpoints/stage1_full_recalibrated/recalibrated_final.pt           |                  0.28 |               -2.8596 |
| ./checkpoints/stage1_full_recalibrated_v2/recalibrated_final.pt        |                  0.28 |               -4.6249 |
| ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt        |                  0.04 |               -5.1028 |
| ./checkpoints/stage1_full/stage1_epoch29.pt                            |                  0.76 |                0.2699 |
| ./checkpoints/stage2_sim_longrun/stage2_epoch29.pt                     |                  0.84 |                0.4949 |
| ./checkpoints/stage2_sim_longrun_recalibrated/recalibrated_final.pt    |                  0.12 |               -4.1435 |
| ./checkpoints/stage2_sim_longrun_recalibrated_v2/recalibrated_final.pt |                  0    |               -5.7004 |


## Published reference numbers (NOT measured by this pipeline)

**VoiceMark, own Table 3**: PESQ=2.2, STOI=0.89, SI-SNR=2.01 dB.

**SafeSpeech, own AudioPure eval**: WER 99.6% -> 85.7% after purification, SIM rises to 0.261. SafeSpeech's own reported AudioPure result (WER, not ACC -- different metric family, listed for context, not a direct row-for-row diff against this project's ACC-based table).
