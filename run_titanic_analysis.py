#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_titanic_analysis.py
Łączy ETL z titanic__data_manipulation.py i wykresy z data_visualization_titanic.py.
Uruchom:
    python run_titanic_analysis.py        # PyCharm / konsola
albo w IPython:
    %run run_titanic_analysis.py
albo w Colab:
    !python run_titanic_analysis.py
lub po prostu:
    import run_titanic_analysis as t
    t.main()
"""

import sys, os, importlib
from pathlib import Path

# ------------- KONFIG -------------
PROJECT_DIR = Path(r"C:\python\Titanic")   # <- dostosuj jeśli repo jest gdzie indziej
sys.path.append(str(PROJECT_DIR))          # umożliwia import własnych modułów

# ------------- IMPORTY WŁASNE -------------
try:
    import titanic__data_manipulation as tdm
except ImportError:
    tdm = None
    print("⚠️  Nie znaleziono titanic__data_manipulation.py – pomijam ETL.")

try:
    import data_visualization_titanic as viz
except ImportError as e:
    raise RuntimeError("Brak pliku data_visualization_titanic.py w PROJECT_DIR") from e

# ------------- FUNKCJE POMOCNICZE -------------
def load_data():
    """Ładuje i zwraca DataFrame Titanica, korzystając z modułu ETL jeśli istnieje."""
    if tdm and hasattr(tdm, "load_titanic_df"):
        return tdm.load_titanic_df()       # Twoja własna funkcja
    else:
        import seaborn as sns
        print("Używam sns.load_dataset('titanic') – brak custom ETL.")
        return sns.load_dataset("titanic")

def preprocess(df):
    """Opcjonalny feature-engineering z titanic__data_manipulation."""
    if tdm and hasattr(tdm, "prepare_data"):
        return tdm.prepare_data(df)
    return df

# ------------- MAIN PIPELINE -------------
def main():
    df = load_data()
    df = preprocess(df)

    print(f"Data shape: {df.shape}. Kolumny: {list(df.columns)}")

    # Wywołujemy wszystkie wykresy z modułu viz
    viz.plot_class_distribution(df)
    viz.plot_survival_by_sex_class(df)
    viz.plot_age_distribution(df)
    viz.plot_fare_by_class(df)
    viz.plot_embarked(df)
    viz.plot_family_size(df)
    viz.plot_corr_heatmap(df)

    # Jeśli chcesz ręcznie wybrać:   viz.plot_xyz(df)

if __name__ == "__main__":
    main()