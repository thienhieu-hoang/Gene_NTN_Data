"""
====================================================================================================
Plot NMSE vs. SNR for Linear Interpolation (LI) & Baseline Methods
====================================================================================================

Dataset: A100_2p18e9_600km_70deg_30kHz
Directory: generatedChan/MATLAB/A100_2p18e9_600km_70deg_30kHz/

Overview:
---------
This script scans all SNR subfolders (e.g. SNR_-10dB, SNR_-5dB, SNR_0dB, SNR_5dB, SNR_10dB, SNR_15dB),
loads `nmse_li` (along with `nmse_ls` and `nmse_prac` if present) from each `matlabNTN.mat`, 
prints a consolidated performance table, and generates high-resolution comparison plots (PNG & PDF).

Outputs Generated:
------------------
- `nmse_li_vs_snr_dB.pdf` / `nmse_li_vs_snr_dB.png`     : NMSE (dB) vs. SNR (dB)
- `nmse_comparison_vs_snr_dB.pdf` / `.png`             : Comparative (LI vs. LS vs. Practical)
- `nmse_vs_snr_summary.csv`                            : Tabulated numerical metrics
- `nmse_vs_snr_summary.mat`                            : MATLAB-compatible summary file

Usage:
------
    python plot_nmse_vs_snr.py
====================================================================================================
"""

import os
import re
import sys
import numpy as np
import scipy.io as sio
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# HELPER FUNCTIONS FOR LOADING
# =============================================================================
def extract_snr_from_folder(folder_name: str):
    """Extract integer or float SNR value from folder string (e.g., 'SNR_-10dB' -> -10.0)."""
    match = re.search(r'SNR_([+-]?\d+(?:\.\d+)?)dB', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'([+-]?\d+(?:\.\d+)?)dB', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'SNR_([+-]?\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def read_mat_field(mat_path: str, field_name: str):
    """Read scalar or array from MATLAB v7 or v7.3 HDF5 file."""
    if not os.path.exists(mat_path):
        return None

    try:
        if h5py.is_hdf5(mat_path):
            with h5py.File(mat_path, 'r') as f:
                if field_name in f:
                    val = f[field_name][()]
                    return float(np.squeeze(val))
        else:
            mat = sio.loadmat(mat_path)
            if field_name in mat:
                val = mat[field_name]
                return float(np.squeeze(val))
    except Exception as e:
        print(f"[Warning] Could not read '{field_name}' from {mat_path}: {e}")
    return None


# =============================================================================
# MAIN PROCESSING & PLOTTING ROUTINE
# =============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print(f"Scanning directory: {script_dir}")
    print("=" * 80)

    # 1. Discover SNR subfolders
    snr_entries = []
    for item in os.listdir(script_dir):
        sub_path = os.path.join(script_dir, item)
        if os.path.isdir(sub_path):
            snr_val = extract_snr_from_folder(item)
            if snr_val is not None:
                mat_file = os.path.join(sub_path, 'matlabNTN.mat')
                if os.path.exists(mat_file):
                    snr_entries.append({
                        'folder': item,
                        'snr': snr_val,
                        'mat_path': mat_file
                    })

    if not snr_entries:
        print("[Error] No SNR folders containing 'matlabNTN.mat' found!")
        return

    # Sort entries by SNR ascending
    snr_entries.sort(key=lambda x: x['snr'])

    snr_list = []
    nmse_li_list = []
    nmse_ls_list = []
    nmse_prac_list = []

    for entry in snr_entries:
        snr = entry['snr']
        mat_path = entry['mat_path']

        li_val = read_mat_field(mat_path, 'nmse_li')
        ls_val = read_mat_field(mat_path, 'nmse_ls')
        prac_val = read_mat_field(mat_path, 'nmse_prac')

        snr_list.append(snr)
        nmse_li_list.append(li_val)
        nmse_ls_list.append(ls_val)
        nmse_prac_list.append(prac_val)

    snr_arr = np.array(snr_list)
    nmse_li_arr = np.array(nmse_li_list, dtype=np.float64)
    nmse_ls_arr = np.array(nmse_ls_list, dtype=np.float64)
    nmse_prac_arr = np.array(nmse_prac_list, dtype=np.float64)

    # Convert to dB scale: 10 * log10(NMSE)
    nmse_li_db = 10.0 * np.log10(np.maximum(nmse_li_arr, 1e-30))
    nmse_ls_db = 10.0 * np.log10(np.maximum(nmse_ls_arr, 1e-30))
    nmse_prac_db = 10.0 * np.log10(np.maximum(nmse_prac_arr, 1e-30))

    # 2. Display Consolidated Table
    print(f"\n{'SNR (dB)':^10} | {'NMSE (LI) Lin':^15} | {'NMSE (LI) dB':^15} | {'NMSE (LS) dB':^15} | {'NMSE (Prac) dB':^15}")
    print("-" * 80)
    for i in range(len(snr_arr)):
        li_s = f"{nmse_li_arr[i]:.6f}" if not np.isnan(nmse_li_arr[i]) else "N/A"
        li_db_s = f"{nmse_li_db[i]:.2f} dB" if not np.isnan(nmse_li_db[i]) else "N/A"
        ls_db_s = f"{nmse_ls_db[i]:.2f} dB" if not np.isnan(nmse_ls_db[i]) else "N/A"
        prac_db_s = f"{nmse_prac_db[i]:.2f} dB" if not np.isnan(nmse_prac_db[i]) else "N/A"
        print(f"{snr_arr[i]:^10.1f} | {li_s:^15} | {li_db_s:^15} | {ls_db_s:^15} | {prac_db_s:^15}")
    print("-" * 80)

    # 3. Plot 1: Dedicated NMSE LI vs. SNR (dB)
    fig1, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax1.plot(snr_arr, nmse_li_db, marker='s', color='#D9531E', lw=2.2, markersize=8, label='Linear Interpolation (LI)', markeredgecolor='black')
    
    for x, y in zip(snr_arr, nmse_li_db):
        ax1.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='#B03A0E')

    ax1.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
    ax1.set_title(r'$\mathbf{NMSE_{LI}}$ vs. SNR (A100 2.18 GHz 600km 70° 30kHz)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(snr_arr)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', frameon=True, fontsize=11)
    fig1.tight_layout()

    out_pdf1 = os.path.join(script_dir, 'nmse_li_vs_snr_dB.pdf')
    out_png1 = os.path.join(script_dir, 'nmse_li_vs_snr_dB.png')
    fig1.savefig(out_pdf1)
    fig1.savefig(out_png1)
    plt.close(fig1)
    print(f"\n[Save] Exported LI NMSE plot -> {out_pdf1}")
    print(f"[Save] Exported LI NMSE plot -> {out_png1}")

    # 4. Plot 2: Comprehensive Multi-Method Comparison (LI vs LS vs Prac)
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.8), dpi=300)
    if not np.all(np.isnan(nmse_ls_db)):
        ax2.plot(snr_arr, nmse_ls_db, marker='^', color='#0072BD', lw=2.0, markersize=7, label='LS Pilot Estimation', markeredgecolor='black', ls='--')
    ax2.plot(snr_arr, nmse_li_db, marker='s', color='#D9531E', lw=2.4, markersize=8, label='Linear Interpolation (LI)', markeredgecolor='black')
    if not np.all(np.isnan(nmse_prac_db)):
        ax2.plot(snr_arr, nmse_prac_db, marker='o', color='#77AC30', lw=2.0, markersize=7, label='Practical Channel Estimation', markeredgecolor='black', ls='-.')

    ax2.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Channel Estimation NMSE Comparison vs. SNR', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(snr_arr)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', frameon=True, fontsize=11)
    fig2.tight_layout()

    out_pdf2 = os.path.join(script_dir, 'nmse_comparison_vs_snr_dB.pdf')
    out_png2 = os.path.join(script_dir, 'nmse_comparison_vs_snr_dB.png')
    fig2.savefig(out_pdf2)
    fig2.savefig(out_png2)
    plt.close(fig2)
    print(f"[Save] Exported Comparative plot -> {out_pdf2}")
    print(f"[Save] Exported Comparative plot -> {out_png2}")

    # 5. Export to CSV & MAT
    csv_path = os.path.join(script_dir, 'nmse_vs_snr_summary.csv')
    with open(csv_path, 'w') as f:
        f.write("SNR_dB,NMSE_LI_linear,NMSE_LI_dB,NMSE_LS_linear,NMSE_LS_dB,NMSE_Prac_linear,NMSE_Prac_dB\n")
        for i in range(len(snr_arr)):
            f.write(f"{snr_arr[i]},{nmse_li_arr[i]},{nmse_li_db[i]},{nmse_ls_arr[i]},{nmse_ls_db[i]},{nmse_prac_arr[i]},{nmse_prac_db[i]}\n")
    print(f"[Save] Exported summary CSV -> {csv_path}")

    mat_path = os.path.join(script_dir, 'nmse_vs_snr_summary.mat')
    sio.savemat(mat_path, {
        'snr_dB': snr_arr,
        'nmse_li_linear': nmse_li_arr,
        'nmse_li_dB': nmse_li_db,
        'nmse_ls_linear': nmse_ls_arr,
        'nmse_ls_dB': nmse_ls_db,
        'nmse_prac_linear': nmse_prac_arr,
        'nmse_prac_dB': nmse_prac_db
    })
    print(f"[Save] Exported summary MAT -> {mat_path}")
    print("\n[Done] All plots and summaries generated successfully.")


if __name__ == '__main__':
    main()
