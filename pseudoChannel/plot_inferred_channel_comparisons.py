"""
Generates matching-style channel visualization plots for H_perfect, H_LS_infer, and H_li
across all SNR subfolders in inferred_dataset/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_standardize.

Outputs:
  - sample_{idx}_magnitude.png
  - sample_{idx}_real.png
for representative samples [1, 34, 67, 100].
"""

import os
import re
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

inferred_root = r"c:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\pseudoChannel\inferred_dataset\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz\LS_Attention_standardize"

def standardize_dims(arr):
    # Target shape: (14 symbols, 132 subcarriers, N)
    shape = arr.shape
    if shape[0] == 14 and shape[1] == 132:
        return arr
    elif shape[1] == 132 and shape[2] == 14:
        return np.transpose(arr, (2, 1, 0))
    elif shape[0] == 132 and shape[1] == 14:
        return np.transpose(arr, (1, 0, 2))
    elif shape[1] == 14 and shape[2] == 132:
        return np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected shape: {shape}")

def main():
    if not os.path.exists(inferred_root):
        print(f"Directory not found: {inferred_root}")
        return

    subdirs = [d for d in os.listdir(inferred_root) if d.startswith("LS_") and os.path.isdir(os.path.join(inferred_root, d))]
    # Sort subdirs by SNR
    def snr_key(name):
        m = re.search(r'LS_([+-]?\d+)dB', name)
        return int(m.group(1)) if m else 999
    subdirs.sort(key=snr_key)

    print(f"Found {len(subdirs)} SNR subdirectories: {subdirs}")

    for folder_name in subdirs:
        folder_path = os.path.join(inferred_root, folder_name)
        mat_path = os.path.join(folder_path, "inferredChannel.mat")
        if not os.path.exists(mat_path):
            print(f"[Skip] {mat_path} not found.")
            continue

        print(f"\n--> Processing: {folder_name} ...")
        m_snr = re.search(r'LS_([+-]?\d+)dB', folder_name)
        snr_val = int(m_snr.group(1)) if m_snr else None

        data = sio.loadmat(mat_path)
        if 'H_perfect' not in data or 'H_LS_infer' not in data:
            print(f"[Skip] Missing H_perfect or H_LS_infer in {folder_name}")
            continue

        H_perf_all = standardize_dims(data['H_perfect'].astype(np.complex128))
        H_infer_all = standardize_dims(data['H_LS_infer'].astype(np.complex128))
        has_li = 'H_li' in data
        if has_li:
            H_li_all = standardize_dims(data['H_li'].astype(np.complex128))

        n_samples = H_perf_all.shape[2]
        if n_samples >= 4:
            sample_indices = np.round(np.linspace(1, min(n_samples, 100), 4)).astype(int)
        else:
            sample_indices = np.arange(1, n_samples + 1)

        for idx in sample_indices:
            idx_0 = idx - 1  # 0-based for python indexing

            # Transpose to (132 subcarriers, 14 symbols) matching MATLAB (.')
            h_perf = H_perf_all[:, :, idx_0].T
            h_infer = H_infer_all[:, :, idx_0].T
            h_diff = np.abs(h_infer - h_perf)
            h_li = H_li_all[:, :, idx_0].T if has_li else None

            sample_nmse = np.sum(np.abs(h_infer - h_perf)**2) / np.sum(np.abs(h_perf)**2)
            sample_nmse_db = 10 * np.log10(sample_nmse) if sample_nmse > 0 else -99.9

            # -------------------------------------------------------------
            # 1. Magnitude Figure (2x2 subplots)
            # -------------------------------------------------------------
            fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
            extent = [1, 14, 132, 1]  # symbol 1..14 (x), subcarrier 1..132 (y)

            # Subplot 1: Target H_perfect
            im0 = axs[0, 0].imshow(np.abs(h_perf), aspect='auto', extent=extent, cmap='viridis')
            plt.colorbar(im0, ax=axs[0, 0])
            axs[0, 0].set_xlabel('OFDM Symbol')
            axs[0, 0].set_ylabel('Subcarrier')
            axs[0, 0].set_title(f'Target $H_{{perfect}}$ (Ground Truth, Sample #{idx})', fontsize=10, fontweight='bold')

            # Subplot 2: Model H_LS_infer
            im1 = axs[0, 1].imshow(np.abs(h_infer), aspect='auto', extent=extent, cmap='viridis')
            plt.colorbar(im1, ax=axs[0, 1])
            axs[0, 1].set_xlabel('OFDM Symbol')
            axs[0, 1].set_ylabel('Subcarrier')
            axs[0, 1].set_title(f'Model $H_{{LS\\_infer}}$ (Inferred, Sample #{idx})', fontsize=10, fontweight='bold')

            # Subplot 3: Baseline H_li
            if has_li:
                im2 = axs[1, 0].imshow(np.abs(h_li), aspect='auto', extent=extent, cmap='viridis')
                plt.colorbar(im2, ax=axs[1, 0])
                axs[1, 0].set_xlabel('OFDM Symbol')
                axs[1, 0].set_ylabel('Subcarrier')
                axs[1, 0].set_title(f'Baseline $H_{{li}}$ (Linear Interpolation, Sample #{idx})', fontsize=10, fontweight='bold')
            else:
                axs[1, 0].axis('off')

            # Subplot 4: Magnitude Error
            im3 = axs[1, 1].imshow(h_diff, aspect='auto', extent=extent, cmap='viridis')
            plt.colorbar(im3, ax=axs[1, 1])
            axs[1, 1].set_xlabel('OFDM Symbol')
            axs[1, 1].set_ylabel('Subcarrier')
            axs[1, 1].set_title(f'Magnitude Error $|H_{{infer}} - H_{{perfect}}|$ (NMSE: {sample_nmse_db:.2f} dB)', fontsize=10, fontweight='bold')

            snr_str = f"SNR {snr_val} dB" if snr_val is not None else folder_name
            fig.suptitle(f'Inferred Channel Magnitude |H| - Sample #{idx} ({snr_str}, NMSE {sample_nmse_db:.2f} dB)',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()

            mag_out = os.path.join(folder_path, f"sample_{idx}_magnitude.png")
            plt.savefig(mag_out, dpi=150)
            plt.close(fig)

            # -------------------------------------------------------------
            # 2. Real Part Figure (2x2 subplots)
            # -------------------------------------------------------------
            fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))

            # Subplot 1: Target H_perfect Real
            im0 = axs[0, 0].imshow(np.real(h_perf), aspect='auto', extent=extent, cmap='viridis')
            plt.colorbar(im0, ax=axs[0, 0])
            axs[0, 0].set_xlabel('OFDM Symbol')
            axs[0, 0].set_ylabel('Subcarrier')
            axs[0, 0].set_title(f'Target $H_{{perfect}}$ (Ground Truth, Sample #{idx})', fontsize=10, fontweight='bold')

            # Subplot 2: Model H_LS_infer Real
            im1 = axs[0, 1].imshow(np.real(h_infer), aspect='auto', extent=extent, cmap='viridis')
            plt.colorbar(im1, ax=axs[0, 1])
            axs[0, 1].set_xlabel('OFDM Symbol')
            axs[0, 1].set_ylabel('Subcarrier')
            axs[0, 1].set_title(f'Model $H_{{LS\\_infer}}$ (Inferred, Sample #{idx})', fontsize=10, fontweight='bold')

            # Subplot 3: Baseline H_li Real
            if has_li:
                im2 = axs[1, 0].imshow(np.real(h_li), aspect='auto', extent=extent, cmap='viridis')
                plt.colorbar(im2, ax=axs[1, 0])
                axs[1, 0].set_xlabel('OFDM Symbol')
                axs[1, 0].set_ylabel('Subcarrier')
                axs[1, 0].set_title(f'Baseline $H_{{li}}$ (Linear Interpolation, Sample #{idx})', fontsize=10, fontweight='bold')
            else:
                axs[1, 0].axis('off')

            # Subplot 4: Real Error
            im3 = axs[1, 1].imshow(np.real(h_infer - h_perf), aspect='auto', extent=extent, cmap='viridis')
            plt.colorbar(im3, ax=axs[1, 1])
            axs[1, 1].set_xlabel('OFDM Symbol')
            axs[1, 1].set_ylabel('Subcarrier')
            axs[1, 1].set_title(f'Real Error $Re(H_{{infer}} - H_{{perfect}})$ (Sample #{idx})', fontsize=10, fontweight='bold')

            fig.suptitle(f'Inferred Channel Real Part Re(H) - Sample #{idx} ({snr_str}, NMSE {sample_nmse_db:.2f} dB)',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()

            real_out = os.path.join(folder_path, f"sample_{idx}_real.png")
            plt.savefig(real_out, dpi=150)
            plt.close(fig)

        print(f"  -> Generated 4 pairs of figures (Magnitude & Real) in: {folder_path}")

    print("\nAll inferred channel comparison plots completed successfully!")

if __name__ == "__main__":
    main()
