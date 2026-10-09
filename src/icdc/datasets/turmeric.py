"""Turmeric FT-Raman Adulteration Dataset Loader (Zenodo Record 6014148).

Data source:
"Real or fake yellow in the vibrant colour craze: Rapid detection of lead chromate in turmeric"
Sara W. Erasmus, Lisanne van Hasselt, Linda M. Ebbinge, Saskia M. van Ruth.
Food Control 121 (2021) 107714. DOI: 10.1016/j.foodcont.2020.107714
Licensed under CC BY-SA (Zenodo: https://zenodo.org/records/6014148).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import urllib.request
import numpy as np
import pandas as pd


ZENODO_EXCEL_URL = (
    "https://zenodo.org/records/6014148/files/"
    "Data_Spices_Task%203.1.5_Rapid%20spectroscopy-based%20screening%20techniques%20for%20spices%20(Raman).xlsx"
)


def download_turmeric_dataset(
    dest_path: str = "data/adulteration_turmeric_raman/turmeric_lead_chromate_raman.xlsx",
) -> Path:
    """Download the turmeric lead chromate Raman dataset from Zenodo if not present."""
    path = Path(dest_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading turmeric Raman dataset from Zenodo to {path}...")
        urllib.request.urlretrieve(ZENODO_EXCEL_URL, path)
        print("Download complete.")
    return path


def load_turmeric_dataset(
    excel_path: str = "data/adulteration_turmeric_raman/turmeric_lead_chromate_raman.xlsx",
    region: str = "fingerprint",  # "fingerprint" (1750-200 cm-1) or "full" (3601-200 cm-1)
) -> Dict[str, Any]:
    """Load and format the turmeric lead chromate Raman dataset.

    Returns:
        dict with keys:
            - 'X_laced': (70, P) ndarray of Raman spectra laced with lead chromate
            - 'y_conc_pct': (70,) ndarray of lead chromate concentration in w/w% [0, 0.1, 0.5, 1, 2.5, 5, 10]
            - 'y_conc_mg_kg': (70,) ndarray of concentration in mg/kg (ppm) [0 to 100,000]
            - 'sample_ids': (70,) list of base turmeric sample identifiers ('Sample 1', ..., 'Sample 15')
            - 'wavenumbers': (P,) ndarray of Raman shift values in cm-1
            - 'X_authentic': (15, P) ndarray of pure authentic turmeric samples
            - 'X_pure_adulterant': (3, P) ndarray of 100% pure lead chromate reference spectra
            - 'trial_series': dict containing the single-batch dilution series (Sample 7, 0% to 30%)
    """
    path = download_turmeric_dataset(excel_path)

    # 1. Load main laced series (Fingerprint region 1750-200 cm-1 is default and most diagnostic)
    sheet_name = "Raw spectra region 1750-200" if region == "fingerprint" else "Raw spectra region 3601-200"
    df_laced = pd.read_excel(path, sheet_name=sheet_name)

    wavenumbers = df_laced.iloc[1, 2:].values.astype(float)
    sample_ids = df_laced.iloc[2:, 0].values.astype(str)
    y_conc_pct = df_laced.iloc[2:, 1].values.astype(float) * 100.0  # Fraction to percentage (0% to 10%)
    y_conc_mg_kg = y_conc_pct * 10000.0  # 1% = 10,000 mg/kg
    x_laced = df_laced.iloc[2:, 2:].values.astype(float)

    # 2. Load authentic turmeric samples (15 commercial samples)
    df_auth = pd.read_excel(path, sheet_name="Raw spectra authentic samples")
    wn_auth = df_auth.iloc[1, 2:].values.astype(float)
    x_auth_raw = df_auth.iloc[2:, 2:].values.astype(float)

    # Interpolate authentic spectra to match exact target wavenumbers
    sort_auth = np.argsort(wn_auth)
    wn_auth_sorted = wn_auth[sort_auth]
    x_auth_sorted = x_auth_raw[:, sort_auth]

    x_auth_interp = np.zeros((len(x_auth_raw), len(wavenumbers)), dtype=np.float32)
    for i in range(len(x_auth_raw)):
        x_auth_interp[i] = np.interp(wavenumbers, wn_auth_sorted, x_auth_sorted[i])

    # 3. Load pure lead chromate reference spectra from Trial sheet
    df_trial = pd.read_excel(path, sheet_name="Trial")
    wn_trial = df_trial.iloc[1, 1:].values.astype(float)
    pure_pb_raw = df_trial[df_trial.iloc[:, 0] == 1].iloc[:, 1:].values.astype(float)

    sort_trial = np.argsort(wn_trial)
    wn_trial_sorted = wn_trial[sort_trial]
    pure_pb_sorted = pure_pb_raw[:, sort_trial]

    x_pure_adulterant = np.zeros((len(pure_pb_raw), len(wavenumbers)), dtype=np.float32)
    for i in range(len(pure_pb_raw)):
        x_pure_adulterant[i] = np.interp(wavenumbers, wn_trial_sorted, pure_pb_sorted[i])

    # 4. Load trial series (dilution curve on single batch Sample 7)
    conc_trial = df_trial.iloc[2:, 0].values.astype(float) * 100.0
    x_trial_raw = df_trial.iloc[2:, 1:].values.astype(float)
    x_trial_interp = np.zeros((len(x_trial_raw), len(wavenumbers)), dtype=np.float32)
    for i in range(len(x_trial_raw)):
        x_trial_interp[i] = np.interp(wavenumbers, wn_trial_sorted, x_trial_raw[i, sort_trial])

    return {
        "X_laced": x_laced.astype(np.float32),
        "y_conc_pct": y_conc_pct.astype(np.float32),
        "y_conc_mg_kg": y_conc_mg_kg.astype(np.float32),
        "sample_ids": sample_ids,
        "wavenumbers": wavenumbers.astype(np.float32),
        "X_authentic": x_auth_interp.astype(np.float32),
        "X_pure_adulterant": x_pure_adulterant.astype(np.float32),
        "trial_series": {
            "concentrations_pct": conc_trial.astype(np.float32),
            "X": x_trial_interp.astype(np.float32),
        },
    }
