# Xenon Scattering Analysis

This project contains Python scripts for analyzing laser scattering signals in Xenon experiments.


# **Module:** `scripts/temp_rms.py`

## **Purpose**

This script processes voltage waveforms from PicoScope and temperature readings from LabJack+TSic sensors for the Xenon scattering experiment. It:

1. Automatically locates PicoScope CSV files and temperature sensor CSV in the specified experiment folder.
2. Calculates the RMS of the voltage signal per second.
3. Interpolates temperature data to match the RMS time points.
4. Outputs:

   * A merged CSV file with RMS and temperature data.
   * A three-panel plot:

     * RMS vs Time
     * Temperature vs Time
     * RMS vs Temperature scatter plot
   * Optional error bars for both RMS and temperature.

---

## **Directory Structure**

```
xenon_scattering_analysis/
├── data/
│   ├── exp01/
│   │   ├── <picoscope_csv...>
│   │   └── sensor_readings_YYYY_MM_DD.csv
│   └── exp02/
├── output/
│   ├── exp01/
│   └── exp02/
├── scripts/
│   └── temp_rms.py
└── README.md
```

---

## **How to Run**

From the project root:

```bash
python3 scripts/temp_rms.py --exp exp01
```

From any directory (explicitly specify project root):

```bash
python3 scripts/temp_rms.py --exp exp01 \
  --repo-root /path/to/xenon_scattering_analysis
```

If automatic temperature file detection fails, specify it manually:

```bash
python3 scripts/temp_rms.py --exp exp02 \
  --temp-file "data/exp02/sensor_readings_2025_08_11_15_11_53.csv"
```

---

## **Command-line Arguments**

| Argument      | Required | Description                                        |
| ------------- | -------- | -------------------------------------------------- |
| `--exp`       | ✅        | Experiment folder name under `data/`               |
| `--repo-root` | ❌        | Path to the repo root (auto-detected if not given) |
| `--overwrite` | ❌        | Overwrite existing outputs                         |
| `--temp-file` | ❌        | Path to temperature CSV (overrides auto-discovery) |

---

## **Output**

The script will create an `output/<exp>/` folder containing:

1. **Merged CSV**: `<exp>_temperature_vs_rms.csv`

   * Columns: `Time`, `RMS`, `Temperature`
   * Optional: `RMS_std` (per-second voltage std), `RMS_se` (standard error)
2. **Three-panel plot**: `<exp>_temp_rms_plots.png`

   * Top: RMS vs Time
   * Middle: Temperature vs Time
   * Bottom: RMS vs Temperature scatter
   * Optional error bars for both RMS and temperature.

---

## **Error Bars**

* **RMS error bars**: Calculated as the standard error of RMS for each second:

  $$
  \text{SE(RMS)} = \frac{\sigma_{\text{voltage}}}{\sqrt{n}}
  $$

  where $n$ = number of samples per second.
* **Temperature error bars**: Currently set to a constant `TEMP_ERROR` value (modifiable in the code to match sensor specs).

---

## **Common Issues**

1. **Temperature file not found**:
   Use `--temp-file` to manually specify the file.
2. **Y-axis range too large**:
   Check if error bar values are too large. For RMS, use standard error instead of raw standard deviation.

---
