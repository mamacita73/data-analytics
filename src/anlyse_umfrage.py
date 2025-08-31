# -*- coding: utf-8 -*-
"""
Analyseverfahren für die Produktkategorien-Umfrage (nur Kategorien)
- Sucht gezielt die Datei 'umfrage_2025.*' in data/raw (keine Unterordner nötig)
- Robust für XLSX/CSV (Delimiter/Encoding-Fallback bei CSV)
- Visualisiert: heute regelmäßig, häufiger/seltener als vor 5 Jahren, Top-5
"""

from pathlib import Path
import re
import argparse
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Pfade ------------------
BASE = Path(__file__).resolve().parents[1]   # Projektbasis (eine Ebene über /src)
DATA = BASE / "data"
RAW = DATA / "raw"                           # <— WICHTIG: kein Unterordner mehr
PROC = DATA / "processed"
FIG  = BASE / "reports" / "figures"
PROC.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ------------------ Konfiguration ------------------
CATEGORY_CANON = [
    "Kleidung / Schuhe",
    "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Lebensmittel / Getränke",
    "Bücher / Medien / Software",
    "Möbel / Wohnaccessoires",
    "Medikamente / Drogerieartikel",
    "Hobby- & Freizeitartikel",
]

CATEGORY_ALIASES: Dict[str, str] = {
    r"^kleidung.*schuhe$": "Kleidung / Schuhe",
    r"^mode$": "Kleidung / Schuhe",
    r"^elektronik.*haushaltsgeräte.*$": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    r"^elektronik.*$": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    r"^lebensmittel.*|^frische lebensmittel.*|^katzenfutter$": "Lebensmittel / Getränke",
    r"^bücher.*|^medien.*|^software$|^dvd.*cd.*videospiele$|^vinyl.*$": "Bücher / Medien / Software",
    r"^möbel.*wohnaccessoires.*$|^möbel$": "Möbel / Wohnaccessoires",
    r"^medikamente.*drogerie.*$": "Medikamente / Drogerieartikel",
    r"^hobby.*freizeit.*$": "Hobby- & Freizeitartikel",
}

# nur die drei nötigen Spalten
COL_PATTERNS = {
    "regular": r"was\s*kaufen\s*sie\s*derzeit\s*regelmäßig\s*online",
    "more":    r"welche.*häufiger\s*online\s*als\s*noch\s*vor\s*fünf\s*jahren",
    "less":    r"welche.*seltener\s*online\s*als\s*noch\s*vor\s*fünf\s*jahren",
}

# ------------------ Helfer ------------------
def find_input_file(cli_path: str | None) -> Path:
    """1) --file nutzen, wenn übergeben; 2) sonst exakt 'umfrage_2025.*' in data/raw wählen."""
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            p = (BASE / p).resolve()
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(f"Angegebene Datei nicht gefunden: {p}")

    if not RAW.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {RAW}")

    # exakt zuerst umfrage_2025.xlsx/csv, sonst Fallback auf 'umfrage' im Namen
    exact = list(RAW.glob("umfrage_2025.xlsx")) + list(RAW.glob("umfrage_2025.csv"))
    if exact:
        return exact[0]
    candidates = list(RAW.glob("*.xlsx")) + list(RAW.glob("*.csv"))
    preferred = [p for p in candidates if "umfrage" in p.stem.lower()]
    if preferred:
        return preferred[0]
    raise FileNotFoundError("Keine Datei 'umfrage_2025.(xlsx|csv)' in data/raw gefunden.")

def read_survey(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype=str)
    else:
        df = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                df = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError(f"CSV konnte nicht geparst werden: {path}")
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print(f"[INFO] Eingelesen: {path}")
    return df

def find_column(df: pd.DataFrame, pattern: str) -> str:
    rx = re.compile(pattern, re.I)
    for c in df.columns:
        if rx.search(c):
            return c
    raise KeyError(f"Spalte für Muster '{pattern}' nicht gefunden.\nErste Spalten: {list(df.columns)[:10]}")

def split_multi(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = re.split(r"[;\n]+", cell)
    drop = {"keine", "keine angabe", "nichts, ich kaufe weniger online"}
    return [p.strip() for p in parts if p and p.strip().lower() not in drop]

def normalize_category(cat: str) -> str:
    if not isinstance(cat, str) or not cat.strip():
        return ""
    key = cat.strip().lower()
    for k in CATEGORY_CANON:
        if key == k.lower():
            return k
    for pat, target in CATEGORY_ALIASES.items():
        if re.match(pat, key, re.I):
            return target
    key2 = re.sub(r"\s+", " ", key).strip()
    for k in CATEGORY_CANON:
        if key2 in k.lower():
            return k
    return "Sonstiges"

def explode_multiselect(df: pd.DataFrame, col: str, value_col: str) -> pd.DataFrame:
    tmp = df[[col]].copy()
    tmp[value_col] = tmp[col].apply(split_multi)
    tmp = tmp.explode(value_col).dropna()
    if tmp.empty:
        return pd.DataFrame(columns=[value_col])
    tmp[value_col] = tmp[value_col].apply(normalize_category)
    return tmp

def percentify(s: pd.Series) -> pd.Series:
    total = s.sum()
    return (100 * s / total).round(1) if total > 0 else s

def save_bar(series: pd.Series, title: str, fname: str, xlabel: str = "Anteil in %"):
    series = percentify(series)
    plt.figure(figsize=(9, 5))
    series.sort_values().plot(kind="barh")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=160)
    plt.close()

def save_diverging_bar(more: pd.Series, less: pd.Series, title: str, fname: str):
    cats = sorted(set(more.index).union(less.index))
    m = percentify(more.reindex(cats).fillna(0))
    l = percentify(less.reindex(cats).fillna(0)) * -1
    y = np.arange(len(cats))
    plt.figure(figsize=(10, max(4, len(cats) * 0.45)))
    bars_more = plt.barh(y, m.values, label="Häufiger gekauft (als vor 5 Jahren)")
    bars_less = plt.barh(y, l.values, label="Seltener gekauft (als vor 5 Jahren)")
    plt.yticks(y, cats)
    plt.axvline(0, linewidth=1, color="black")
    plt.title(title)
    plt.xlabel("Anteil in % (links: seltener, rechts: häufiger)")
    plt.legend(loc="lower right")   # <-- Legende hinzufügen
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=160)
    plt.close()


# ------------------ Hauptlauf ------------------
def main():
    parser = argparse.ArgumentParser(description="Analyse Umfrage 2025 (nur Kategorien)")
    parser.add_argument("--file", "-f", help="Optional: expliziter Pfad zu umfrage_2025.(xlsx|csv)")
    args = parser.parse_args()

    input_file = find_input_file(args.file)
    df = read_survey(input_file)

    col_regular = find_column(df, COL_PATTERNS["regular"])
    col_more    = find_column(df, COL_PATTERNS["more"])
    col_less    = find_column(df, COL_PATTERNS["less"])

    base = df[[col_regular, col_more, col_less]].copy()
    base.rename(columns={
        col_regular: "Regelmäßig",
        col_more:    "Häufiger_als_vor_5J",
        col_less:    "Seltener_als_vor_5J",
    }, inplace=True)

    # 1) Heute regelmäßig
    reg = explode_multiselect(base, "Regelmäßig", "Kategorie")
    reg_counts = reg["Kategorie"].value_counts()
    reg_counts.to_csv(PROC / "regelmaessig_counts.csv", encoding="utf-8")
    save_bar(reg_counts, "Heute regelmäßig online gekaufte Kategorien", "01_regelmaessig.png")

    # 2) Veränderung vs. vor 5 Jahren
    more = explode_multiselect(base, "Häufiger_als_vor_5J", "Kategorie")
    less = explode_multiselect(base, "Seltener_als_vor_5J", "Kategorie")
    more_counts = more["Kategorie"].value_counts()
    less_counts = less["Kategorie"].value_counts()
    (PROC / "veraenderung").mkdir(parents=True, exist_ok=True)
    more_counts.to_csv(PROC / "veraenderung/haeufiger_counts.csv", encoding="utf-8")
    less_counts.to_csv(PROC / "veraenderung/seltener_counts.csv", encoding="utf-8")
    save_diverging_bar(more_counts, less_counts, "Veränderung gegenüber vor 5 Jahren", "02_veraenderung_diverging.png")

    # 3) Top-5
    save_bar(reg_counts.head(5),  "Top-5: regelmäßig online gekauft", "05_top5_regelmaessig.png")
    save_bar(more_counts.head(5), "Top-5: häufiger als vor 5 Jahren", "06_top5_haeufiger.png")
    save_bar(less_counts.head(5), "Top-5: seltener als vor 5 Jahren", "07_top5_seltener.png")

    # 4) Summary in %
    summary = pd.DataFrame({
        "Regelmäßig_%": percentify(reg_counts),
        "Häufiger_%":    percentify(more_counts),
        "Seltener_%":    percentify(less_counts),
    }).fillna(0).sort_values("Regelmäßig_%", ascending=False)
    summary.round(1).to_csv(PROC / "summary_kategorien_pct.csv", encoding="utf-8")

    print("[OK] Ergebnisse gespeichert in:")
    print(f" - Tabellen: {PROC}")
    print(f" - Grafiken: {FIG}")

if __name__ == "__main__":
    main()
