# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`DUR100__A100_2p18e9_600km_30kHz_LSSequence`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.456765 | 0.453162 | 0.450610 |
| -5.0 | 0.409113 | 0.411176 | 0.403962 |
| 0.0 | 0.323676 | 0.326383 | 0.316637 |
| 5.0 | 0.210719 | 0.221973 | 0.204220 |
| 10.0 | 0.102848 | 0.128596 | 0.098163 |
| 15.0 | 0.033314 | 0.067587 | 0.031162 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -0.02 dB | -6.02 dB | -7.76 dB |
| -5.0 | -4.97 dB | -7.04 dB | -12.07 dB |
| 0.0 | -10.05 dB | -9.12 dB | -16.22 dB |
| 5.0 | -14.99 dB | -10.17 dB | -21.14 dB |
| 10.0 | -19.98 dB | -11.64 dB | -26.60 dB |
| 15.0 | -24.79 dB | -12.85 dB | -31.37 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.2531 | 0.2651 | 0.5560 |
| -5.0 | 0.5146 | 0.4317 | 0.8201 |
| 0.0 | 0.7605 | 0.6853 | 0.9325 |
| 5.0 | 0.8891 | 0.7585 | 0.9792 |
| 10.0 | 0.9626 | 0.8300 | 0.9933 |
| 15.0 | 0.9850 | 0.8776 | 0.9974 |
