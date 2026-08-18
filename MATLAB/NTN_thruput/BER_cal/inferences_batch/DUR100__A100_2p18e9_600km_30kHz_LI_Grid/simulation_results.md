# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`DUR100__A100_2p18e9_600km_30kHz_LI_Grid`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.456765 | 0.453500 | 0.450610 |
| -5.0 | 0.409113 | 0.409939 | 0.403962 |
| 0.0 | 0.323676 | 0.319092 | 0.316637 |
| 5.0 | 0.210719 | 0.208523 | 0.204220 |
| 10.0 | 0.102848 | 0.105012 | 0.098163 |
| 15.0 | 0.033314 | 0.037470 | 0.031162 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -0.02 dB | -5.74 dB | -7.76 dB |
| -5.0 | -4.97 dB | -7.54 dB | -12.07 dB |
| 0.0 | -10.05 dB | -13.05 dB | -16.22 dB |
| 5.0 | -14.99 dB | -16.49 dB | -21.14 dB |
| 10.0 | -19.98 dB | -18.37 dB | -26.60 dB |
| 15.0 | -24.79 dB | -20.75 dB | -31.37 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.2531 | 0.2603 | 0.5560 |
| -5.0 | 0.5146 | 0.4421 | 0.8201 |
| 0.0 | 0.7605 | 0.8523 | 0.9325 |
| 5.0 | 0.8891 | 0.9327 | 0.9792 |
| 10.0 | 0.9626 | 0.9577 | 0.9933 |
| 15.0 | 0.9850 | 0.9751 | 0.9974 |
