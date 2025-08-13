#!/usr/bin/env python3
"""
temp_rms: Compute RMS of scattered-light voltage vs time and join with temperature.

Refactor goals:
- Live inside <repo_root>/scripts/temp_rms.py
- Read all PicoScope CSVs and a temperature CSV from <repo_root>/data/<experiment>/
- Write CSV and plots to <repo_root>/output/<experiment>/
- No hard-coded filenames; discover files by pattern.
- Safe to run from anywhere (uses --repo-root or auto-detect by walking up).

Usage examples (from the repo root):
    python scripts/temp_rms.py --exp exp01
    python scripts/temp_rms.py --exp exp01 --overwrite

Or from any directory:
    python /path/to/repo/scripts/temp_rms.py --exp exp01 --repo-root /path/to/repo
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Filesystem helpers
# --------------------------
def find_repo_root(explicit: str | None) -> Path:
    """Resolve the repo root.
    Priority:
        1) --repo-root argument if provided
        2) Walk up from this script file to find a directory containing 'data' and 'output'
    """
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "data").exists():
            raise FileNotFoundError(f"'data' not found under provided repo-root: {root}")
        return root

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "data").exists() and (parent / "output").exists():
            return parent
    raise FileNotFoundError("Could not auto-detect repo root. Pass --repo-root.")


def ensure_out_dir(root: Path, exp: str) -> Path:
    out = root / "output" / exp
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------
# IO: discover and load data
# --------------------------
def discover_files(data_dir: Path) -> Tuple[List[Path], Path]:
    """Find PicoScope CSVs and the temperature CSV under an experiment directory.
    Heuristics:
      - PicoScope files: *.csv excluding ones containing 'sensor' or 'temperature_vs_rms'
      - Temperature file: first CSV whose header contains 'TSic' or 'AIN0' or 'Temperature'
    """
    all_csvs = sorted(data_dir.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    # A quick sniff to identify the temp file by header contents
    temp_file: Path | None = None
    pico_files: List[Path] = []
    for p in all_csvs:
        # Avoid obviously-derived or merged result files
        if "temperature_vs_rms" in p.name.lower():
            continue
        # Try to read a small chunk to identify columns
        try:
            # PicoScope exports typically have ~10 header rows; try both options
            try_rows = [0, 10]
            sniffed = None
            for skip in try_rows:
                try:
                    sniffed = pd.read_csv(p, nrows=5, skiprows=skip)
                    break
                except Exception:
                    continue
            if sniffed is None:
                # Give up and treat as pico for now
                pico_files.append(p)
                continue

            cols = [c.lower() for c in sniffed.columns]
            joined = " ".join(cols)
            if any(key in joined for key in ["tsic", "ain0", "temperature"]):
                temp_file = temp_file or p
            else:
                # likely a two-column PicoScope waveform
                pico_files.append(p)
        except Exception:
            # If unreadable, assume it's PicoScope (header-heavy), we'll try with skiprows=10 later
            pico_files.append(p)

    if not pico_files:
        raise FileNotFoundError(f"No PicoScope waveform CSVs found in {data_dir}")
    if temp_file is None:
        # fall back to a file named like 'sensor_readings' if present
        for p in all_csvs:
            if "sensor" in p.name.lower():
                temp_file = p
                break
    if temp_file is None:
        raise FileNotFoundError("Could not find a temperature CSV (looked for 'TSic/Temperature' columns).")

    return pico_files, temp_file


def load_voltage_data(file_paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate PicoScope CSV files. Each CSV has two columns: time (s), voltage (V).
    We default to skiprows=10 to skip PicoScope's header, and fall back to 0 if needed.
    """
    dfs = []
    for fp in file_paths:
        read_ok = False
        for skip in (10, 0):
            try:
                df = pd.read_csv(fp, skiprows=skip)
                if df.shape[1] >= 2:
                    # Keep only first two columns; normalize names
                    df = df.iloc[:, :2]
                    df.columns = ["Time (s)", "Voltage (V)"]
                    dfs.append(df)
                    read_ok = True
                    break
            except Exception:
                continue
        if not read_ok:
            raise ValueError(f"Failed to parse PicoScope CSV: {fp}")
    return pd.concat(dfs, ignore_index=True)


def load_temperature_data(fp: Path) -> pd.DataFrame:
    """Load temperature log CSV with fixed columns: 'Time (s)' and 'TSic AIN0 (°C)'."""
    df = pd.read_csv(fp)  # 不跳过任何行
    df.columns = [c.strip() for c in df.columns]
    
    if "Time (s)" not in df.columns or "TSic AIN0 (°C)" not in df.columns:
        raise ValueError(f"Expected 'Time (s)' and 'TSic AIN0 (°C)' in columns, got: {df.columns.tolist()}")

    df = df[["Time (s)", "TSic AIN0 (°C)"]].copy()
    df.columns = ["Time", "Temperature"]

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df = df.dropna()

    return df



# --------------------------
# Analysis
# --------------------------
def compute_rms(voltage_df):
    sec = voltage_df["Time (s)"].astype(int)
    grouped = voltage_df.assign(Second=sec).groupby("Second")["Voltage (V)"]

    rms_df = grouped.agg(
        RMS=lambda x: np.sqrt(np.mean(np.square(x))),
        RMS_std=lambda x: np.std(x),
        N='count'
    ).reset_index()

    rms_df.rename(columns={"Second": "Time"}, inplace=True)


    rms_df["RMS_se"] = rms_df["RMS_std"] / np.sqrt(rms_df["N"])
    return rms_df



def interpolate_temperature(rms_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate temperature to the RMS time grid and merge."""
    # Limit to valid, sorted time
    temp_df = temp_df.sort_values("Time")
    interp = np.interp(rms_df["Time"].values, temp_df["Time"].values, temp_df["Temperature"].values)
    merged = rms_df.copy()
    merged["Temperature"] = interp
    return merged


# --------------------------
# Output
# --------------------------
def save_outputs(merged: pd.DataFrame, out_dir: Path, exp: str) -> Tuple[Path, Path]:
    csv_path = out_dir / f"{exp}_temperature_vs_rms.csv"
    plot_path = out_dir / f"{exp}_temp_rms_plots.png"

    merged.to_csv(csv_path, index=False)

    # Plot similar to user's sample: 3 panels
    fig = plt.figure(figsize=(16, 12))

    # Top: RMS vs time
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(merged["Time"], merged["RMS"], label="RMS of Voltage", color="blue")
    ax1.set_title("RMS of Scattered Light Signal vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMS (V)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()
    
    ax1.errorbar(
        merged["Time"], merged["RMS"],
        yerr=merged["RMS_se"],
        fmt='-', color="blue", ecolor="lightblue", capsize=2
    )



    # Middle: Temperature vs time
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(merged["Time"], merged["Temperature"], label="Interpolated Temperature", color="red")
    ax2.set_title("Temperature vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    TEMP_ERROR = 0.005

    ax2.errorbar(
        merged["Time"], merged["Temperature"],
        yerr=TEMP_ERROR,
        fmt='-', color="red", ecolor="lightcoral",
        elinewidth=1, capsize=2,
        label="Interpolated Temperature ± error"
    )


    # Bottom: RMS vs Temperature scatter
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.scatter(merged["Temperature"], merged["RMS"], s=12, color="orange")
    ax3.set_title("RMS vs Temperature")
    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("RMS (V)")
    ax3.grid(True, which="both", alpha=0.3)
    
    

    plt.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return csv_path, plot_path


# --------------------------
# CLI
# --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute RMS from PicoScope CSVs and merge with temperature logs.")
    p.add_argument("--exp", required=True, help="Experiment folder name under data/, e.g. 'exp01'")
    p.add_argument("--repo-root", default=None, help="Path to xenon_scattering_analysis repo root. If omitted, auto-detect.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist.")
    p.add_argument("--temp-file", help="Path to temperature CSV file (overrides auto-discovery)")
    return p.parse_args()


def main():
    args = parse_args()
    root = find_repo_root(args.repo_root)
    data_dir = (root / "data" / args.exp)
    if not data_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {data_dir}")

    out_dir = ensure_out_dir(root, args.exp)

    pico_files, temp_file = discover_files(data_dir)

    voltage_df = load_voltage_data(pico_files)
    temp_df = load_temperature_data(temp_file)

    rms_df = compute_rms(voltage_df)
    merged_df = interpolate_temperature(rms_df, temp_df)

    csv_path = out_dir / f"{args.exp}_temperature_vs_rms.csv"
    plot_path = out_dir / f"{args.exp}_temp_rms_plots.png"
    if (csv_path.exists() or plot_path.exists()) and not args.overwrite:
        print(f"Outputs already exist in {out_dir}. Use --overwrite to replace.")
    csv_path, plot_path = save_outputs(merged_df, out_dir, args.exp)
    
    if args.temp_file:
        temp_file = (root / args.temp_file) if not Path(args.temp_file).is_absolute() else Path(args.temp_file)
    else:
        pico_files, temp_file = discover_files(data_dir)

    print("Analysis complete. Output saved to:")
    print(f"- {csv_path}")
    print(f"- {plot_path}")
    


if __name__ == "__main__":
    main()
