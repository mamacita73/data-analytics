# src/analyse_kategorien.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ------------------ Pfade ------------------
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
FIG = BASE / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ------------------ 1) Zielkategorien (harmonisiert) ------------------
KANON = [
    "Kleidung / Schuhe",
    "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Lebensmittel / Getränke",
    "Bücher / Medien / Software",
    "Medikamente / Drogerieartikel",
    "Hobby- & Freizeitartikel",
    "Möbel / Wohnaccessoires",
]

# ------------------ 2) Mapping Statista -> Zielkategorien ------------------
# (Nur 2021, 2022, 2024. 2023 und Umfrage-Daten sind entfernt.)
MAP_STATISTA = {
    # 2021
    "Fashion & Accessoires": "Kleidung / Schuhe",
    "Freizeit": "Hobby- & Freizeitartikel",
    "Kosmetik & Drogerie": "Medikamente / Drogerieartikel",
    "Medikamente & Gesundheit": "Medikamente / Drogerieartikel",
    "Wohnen & Einrichten": "Möbel / Wohnaccessoires",
    "Consumer Electronics/Elektrogeräte": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Heimwerken & Garten": "Hobby- & Freizeitartikel",
    "Sport- und Outdoorausrüstung": "Hobby- & Freizeitartikel",
    "Lebensmittel": "Lebensmittel / Getränke",
    "Getränke": "Lebensmittel / Getränke",

    # 2022
    "Kleidung, Schuhe, Accessoires": "Kleidung / Schuhe",
    "Bücher, Hörbücher": "Bücher / Medien / Software",
    "Elektrozubehör": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Medikamente": "Medikamente / Drogerieartikel",
    "Kosmetik, Parfum und Pflegeprodukte": "Medikamente / Drogerieartikel",
    "Möbel, Wohnen und Dekoration": "Möbel / Wohnaccessoires",
    "Elektronische Haushaltsgeräte": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Lebensmittel, Getränke": "Lebensmittel / Getränke",
    "Heimwerkerbedarf": "Hobby- & Freizeitartikel",
    "Spielzeug / Spiele": "Hobby- & Freizeitartikel",
    "Smartphones / Handys": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Musik / Filme auf CD / DVD / Blu-ray": "Bücher / Medien / Software",
    "Computer: Desktop-PC, Tablet, Laptop": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Gartengeräte": "Hobby- & Freizeitartikel",
    "Software": "Bücher / Medien / Software",
    "Unterhaltungselektronik": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",

    # 2024
    "Bekleidung": "Kleidung / Schuhe",
    "Schuhe": "Kleidung / Schuhe",
    "Accessoires": "Kleidung / Schuhe",
    "Drogerie & Gesundheit": "Medikamente / Drogerieartikel",
    "Kosmetik & Körperpflege": "Medikamente / Drogerieartikel",
    "Bücher, Filme, Musik & Spiele (keine Downloads)": "Bücher / Medien / Software",
    "Unterhaltungselektronik (z. B. Fernseher, Smartphones)": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Haushaltsgeräte": "Elektronik (z. B. Smartphones, Haushaltsgeräte)",
    "Lebensmittel und Getränke (ausgenommen Lieferungen von Restaurants)": "Lebensmittel / Getränke",
    "Möbel & Haushaltswaren": "Möbel / Wohnaccessoires",
    "Spielzeug & Babyprodukte": "Hobby- & Freizeitartikel",
    "Sport- & Outdoor-Artikel": "Hobby- & Freizeitartikel",
    "DIY, Heimwerker- und Gartenbedarf": "Hobby- & Freizeitartikel",
    "Hobby & Schreibwaren": "Hobby- & Freizeitartikel",
}

# ------------------ Hilfsfunktionen ------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    CSV robust einlesen:
    - probiert mehrere Encodings (utf-8-sig, cp1252, latin1)
    - erkennt Trennzeichen automatisch
    """
    encodings = ["utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError as e:
            last_err = e
            continue
    try:
        return pd.read_csv(path, encoding="latin1", sep=None, engine="python")
    except Exception as e:
        raise RuntimeError(f"CSV konnte nicht gelesen werden ({path}). Letzter Fehler: {last_err or e}") from e

def norm_text(s: str) -> str:
    """Normalisiert Strings für das Mapping."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("&", "und").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    # häufige Varianten/Typo-Fixes
    s = s.replace("accressoires", "accessoires").replace("hi tech", "high tech")
    return s

MAP_NORM = {norm_text(k): v for k, v in MAP_STATISTA.items()}

def pick_statista_cols(df: pd.DataFrame) -> tuple[str, str | None]:
    """Findet Spalten für (Kategorie, Wert)."""
    cat_col = None
    val_col = None
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for key, c in normalized.items():
        if key in {"category","kategorie","warengruppe","produktkategorie","bereich"}:
            cat_col = c
        if key in {"value","wert","prozent","anteil","share","%","rate","anzahl"}:
            val_col = c
    if cat_col is None:
        raise ValueError("Statista-Datei: 'category/kategorie/warengruppe' Spalte nicht gefunden.")
    return cat_col, val_col

def clean_value_col(series: pd.Series) -> pd.Series:
    """Bereinigt Prozentzeichen und Dezimal-Komma und konvertiert zu float, NAs -> 0."""
    return (series.astype(str)
                 .str.replace("%","", regex=False)
                 .str.replace(",",".", regex=False)
                 .str.strip()
                 .pipe(pd.to_numeric, errors="coerce")
                 .fillna(0.0))

def map_category_to_kanon(cat: pd.Series) -> pd.Series:
    return cat.map(lambda x: MAP_NORM.get(norm_text(x), "IGNORE"))

# ------------------ 3) Loader für Statista-CSV ------------------
def load_statista_csv(year: int, fname: str) -> pd.DataFrame:
    df = read_csv_robust(RAW / fname)
    cat_col, val_col = pick_statista_cols(df)

    df = df[[cat_col] + ([val_col] if val_col else [])].copy()
    df.rename(columns={cat_col: "source_category"}, inplace=True)

    if val_col is None:
        df["value"] = pd.NA
    else:
        df["value"] = clean_value_col(df[val_col])

    df["year"] = year
    df["Kategorie"] = map_category_to_kanon(df["source_category"]).fillna("IGNORE")
    df = df[df["Kategorie"] != "IGNORE"].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # Mehrfach-Splits (z. B. Bekleidung/Schuhe/Accessoires) konsolidieren
    out = df.groupby(["year", "Kategorie"], as_index=False)["value"].sum()
    return out

# ------------------ 4) Statista-Jahre laden & mergen (nur 2021, 2022, 2024) ------------------
stat_2021 = load_statista_csv(2021, "statista_2021.csv")
stat_2022 = load_statista_csv(2022, "statista_2022.csv")
stat_2024 = load_statista_csv(2024, "statista_2024.csv")

stat_all = pd.concat([stat_2021, stat_2022, stat_2024], ignore_index=True)

# ------------------ 5) Harmonisierung & Exporte ------------------
# Pivot → Kategorien x Jahre
stat_pivot = stat_all.pivot(index="Kategorie", columns="year", values="value").fillna(0.0)
stat_pivot = stat_pivot.reindex(KANON).fillna(0.0)

# Exporte
(stat_all.sort_values(["year","Kategorie"])
         .to_csv(OUT / "statista_long_2021_2022_2024.csv", index=False, encoding="utf-8"))
(stat_pivot.reset_index()
          .to_csv(OUT / "statista_harmonisiert_2021_2022_2024.csv", index=False, encoding="utf-8"))

# ------------------ 6) Plot-Helfer ------------------
def plot_year_niveau(year: int, df_pivot: pd.DataFrame, fig_dir: Path) -> None:
    """Erstellt eine horizontale Rangliste (wie bei 2024) für das angegebene Jahr."""
    if year not in df_pivot.columns:
        return  # Jahr nicht vorhanden
    plt.figure(figsize=(10, 6))
    values = df_pivot[year].sort_values(ascending=True)
    plt.barh(values.index, values.values)
    plt.title(f"Statista {year}: Anteile je Kategorie (harmonisiert)")
    plt.xlabel("Anteil / in %")
    plt.tight_layout()
    plt.savefig(fig_dir / f"statista_{year}_niveau.png", dpi=200)
    plt.close()

# ------------------ 7) Plots ------------------
# 7.1 Zeitreihe 2021/2022/2024 (Gruppenbalken)
plt.figure(figsize=(12, 6))
stat_pivot.plot(kind="bar", ax=plt.gca())
plt.title("Statista (harmonisiert): 2021, 2022, 2024")
plt.ylabel("Anteil / in % (laut Quelle)")
plt.xlabel("Kategorie")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIG / "statista_zeitreihe_2021_2022_2024.png", dpi=200)
plt.close()

# 7.2 Niveau-Plots je Jahr (horizontal wie im Screenshot)
for yr in [2021, 2022, 2024]:
    plot_year_niveau(yr, stat_pivot, FIG)

print("Fertig. Dateien unter:")
print(f"- {OUT/'statista_long_2021_2022_2024.csv'}")
print(f"- {OUT/'statista_harmonisiert_2021_2022_2024.csv'}")
print(f"- {FIG/'statista_zeitreihe_2021_2022_2024.png'}")
print(f"- {FIG/'statista_2021_niveau.png'}")
print(f"- {FIG/'statista_2022_niveau.png'}")
print(f"- {FIG/'statista_2024_niveau.png'}")
