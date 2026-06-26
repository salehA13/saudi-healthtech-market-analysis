"""
Saudi Healthtech Market Analysis — Chart Generation
====================================================

Generates 7 visualizations for a market sizing playground project.

Usage:
    python src/generate_all_charts.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


COLORS: dict[str, str] = {
    "primary":   "#1B2A4A",
    "secondary": "#2E5090",
    "accent":    "#E8792B",
    "accent2":   "#D4A843",
    "light":     "#F0F4F8",
    "success":   "#2D9A4E",
    "text":      "#1D2530",
    "muted":     "#7A8B9C",
    "bg":        "#FFFFFF",
    "grid":      "#E8ECF0",
}

TIER_COLORS: list[str] = ["#1B2A4A", "#2E5090", "#5B8AC4", "#A3C4E8"]
SCENARIO_COLORS: list[str] = ["#2D9A4E", "#2E5090", "#E8792B"]
REGION_COLORS: list[str] = [
    COLORS["primary"], COLORS["secondary"], "#5B8AC4",
    COLORS["accent2"], COLORS["muted"],
]

CHART_DPI = 200
CHART_FORMAT = "png"

HEALTHCARE_BASE_VALUE_SAR_B = 67.2
HEALTHCARE_CAGR = 0.067
PROJECTION_START_YEAR = 2023
PROJECTION_END_YEAR = 2030

TIER_LABELS = [
    "Tier 1\n(Large Groups)", "Tier 2\n(Regional Chains)",
    "Tier 3\n(Single Site)", "Tier 4\n(Small/Specialty)",
]
TIER_COUNTS = [15, 30, 45, 35]
TIER_AVG_BEDS = [300, 150, 100, 35]
TIER_AVG_REVENUE_SAR_M = [800, 250, 150, 60]

PRICING_PER_ENCOUNTER = [3.2, 12.1, 26.4, 43.8, 58.0]
PRICING_PER_BED = [2.8, 9.5, 20.1, 33.2, 43.0]
PRICING_ENTERPRISE = [5.5, 16.8, 34.2, 55.0, 72.0]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS["grid"],
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.6,
    "grid.linewidth": 0.5,
})


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_market_data() -> dict[str, Any]:
    with open(os.path.join(DATA_DIR, "market_data.json")) as f:
        return json.load(f)


DATA = load_market_data()


def _save_chart(fig: plt.Figure, filename: str) -> None:
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=CHART_DPI,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 1: TAM / SAM / SOM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_tam_sam_som() -> None:
    sizing = DATA["market_sizing"]
    fig, ax = plt.subplots(figsize=(10, 7))

    labels_data = [
        (f"TAM\nSAR {sizing['tam_sar_b']}B", "#D6E4F0"),
        (f"SAM\nSAR {sizing['sam_sar_m']/1000:.1f}B", "#5B8AC4"),
        (f"SOM\nSAR {sizing['som_year5_sar_m']}M", "#1B2A4A"),
    ]

    for i, (_, color) in enumerate(labels_data):
        radius = 0.3 + (2 - i) * 0.25
        ax.add_patch(plt.Circle((0.5, 0.45), radius, color=color, alpha=0.85 - i * 0.1))

    ax.text(0.5, 0.45, f"SOM\nSAR {sizing['som_year5_sar_m']}M\n(Year 5 Target)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(0.5, 0.78, f"SAM — SAR {sizing['sam_sar_m']/1000:.1f}B",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(0.5, 0.05, f"TAM — SAR {sizing['tam_sar_b']}B",
            ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["primary"])

    descriptions = [
        (0.88, "Total addressable market\nacross KSA healthcare"),
        (0.72, "Serviceable segment —\nprivate sector hospitals"),
        (0.45, "Obtainable within\n5-year horizon"),
    ]
    for y, desc in descriptions:
        ax.annotate(desc, xy=(0.82, y), fontsize=9, color=COLORS["muted"],
                    ha="left", va="center")

    ax.set_xlim(-0.1, 1.4)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Market Sizing: TAM / SAM / SOM", fontsize=16, fontweight="bold",
                 color=COLORS["primary"], y=0.97)
    ax.text(0.5, 0.98, "Saudi Healthtech — Illustrative Analysis",
            ha="center", va="top", fontsize=11, color=COLORS["muted"], transform=ax.transAxes)
    plt.tight_layout()
    _save_chart(fig, "tam_sam_som.png")
    print("  ✓ TAM/SAM/SOM chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 2: Healthcare Market Growth Projection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_market_growth() -> None:
    years = list(range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1))
    values = [HEALTHCARE_BASE_VALUE_SAR_B * (1 + HEALTHCARE_CAGR) ** (y - PROJECTION_START_YEAR)
              for y in years]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(years, values, color=COLORS["secondary"], width=0.6,
                  edgecolor="white", linewidth=0.5, zorder=3)
    bars[-1].set_color(COLORS["accent"])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"SAR {val:.1f}B", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=COLORS["text"])

    ax.annotate(f"{HEALTHCARE_CAGR * 100:.1f}% CAGR", xy=(2026.5, 82),
                fontsize=14, fontweight="bold", color=COLORS["accent"], ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["light"],
                          edgecolor=COLORS["accent"], alpha=0.9))

    ax.set_ylabel("Market Size (SAR Billion)", fontweight="bold")
    ax.set_title("Saudi Healthcare Market Projection (2023–2030)",
                 fontsize=14, fontweight="bold", color=COLORS["primary"], pad=15)
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(years)
    ax.text(0.0, -0.12,
            "Source: MoH Statistical Yearbook, Frost & Sullivan (2024)",
            transform=ax.transAxes, fontsize=8, color=COLORS["muted"])
    plt.tight_layout()
    _save_chart(fig, "market_growth.png")
    print("  ✓ Market growth chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 3: Hospital Segmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_hospital_segmentation() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars = ax1.barh(TIER_LABELS, TIER_COUNTS, color=TIER_COLORS, height=0.6, edgecolor="white")
    for bar, val in zip(bars, TIER_COUNTS):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val} hospitals", va="center", fontsize=11, fontweight="bold",
                 color=COLORS["text"])
    ax1.set_xlabel("Number of Hospitals", fontweight="bold")
    ax1.set_title("Hospital Count by Tier", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax1.set_xlim(0, max(TIER_COUNTS) * 1.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.invert_yaxis()

    x_pos = list(range(len(TIER_LABELS)))
    ax2.scatter(x_pos, TIER_AVG_REVENUE_SAR_M,
                s=[b * 5 for b in TIER_AVG_BEDS],
                c=TIER_COLORS, alpha=0.8, edgecolors="white", linewidth=2, zorder=3)
    for i, (rev, beds) in enumerate(zip(TIER_AVG_REVENUE_SAR_M, TIER_AVG_BEDS)):
        ax2.text(i, rev + 30, f"SAR {rev}M\n({beds} beds avg)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLORS["text"])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(TIER_LABELS, fontsize=9)
    ax2.set_ylabel("Avg. Annual Revenue (SAR M)", fontweight="bold")
    ax2.set_title("Revenue & Scale by Tier", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax2.set_ylim(0, max(TIER_AVG_REVENUE_SAR_M) * 1.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(f"Private Hospital Segmentation — Saudi Arabia ({sum(TIER_COUNTS)} Hospitals)",
                 fontsize=15, fontweight="bold", color=COLORS["primary"], y=1.02)
    plt.tight_layout()
    _save_chart(fig, "hospital_segmentation.png")
    print("  ✓ Hospital segmentation chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 4: Regional Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_regional_distribution() -> None:
    rd = DATA["regional_distribution"]
    regions = list(rd.keys())
    hospitals = [rd[r]["hospitals"] for r in regions]
    pcts = [rd[r]["pct"] for r in regions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    bars = ax1.bar(regions, hospitals, color=REGION_COLORS, width=0.6,
                   edgecolor="white", zorder=3)
    for bar, val, pct in zip(bars, hospitals, pcts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{val}\n({pct}%)", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=COLORS["text"])
    ax1.set_ylabel("Number of Private Hospitals", fontweight="bold")
    ax1.set_title("Hospitals by Region", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax1.set_ylim(0, max(hospitals) * 1.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    wedges, texts, autotexts = ax2.pie(
        hospitals, labels=regions, colors=REGION_COLORS,
        autopct="%1.0f%%", startangle=90,
        pctdistance=0.75, wedgeprops=dict(width=0.45),
    )
    for t in autotexts:
        t.set_fontsize(10); t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(9)
    ax2.set_title("Market Concentration", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax2.text(0, 0, f"{sum(hospitals)}\nHospitals", ha="center", va="center",
             fontsize=14, fontweight="bold", color=COLORS["primary"])
    fig.suptitle("Regional Distribution of Private Hospitals", fontsize=15,
                 fontweight="bold", color=COLORS["primary"], y=1.02)
    plt.tight_layout()
    _save_chart(fig, "regional_distribution.png")
    print("  ✓ Regional distribution chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 5: Pricing Model Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_pricing_scenarios() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    x = np.arange(len(year_labels))
    width = 0.25

    models = [
        (PRICING_PER_ENCOUNTER, "Per-Encounter", SCENARIO_COLORS[0]),
        (PRICING_PER_BED, "Per-Bed", SCENARIO_COLORS[1]),
        (PRICING_ENTERPRISE, "Enterprise License", SCENARIO_COLORS[2]),
    ]

    for i, (data, label, color) in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data, width, label=label, color=color,
                      edgecolor="white", zorder=3)
        ax.text(bars[-1].get_x() + bars[-1].get_width() / 2,
                bars[-1].get_height() + 1, f"SAR {data[-1]}M",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLORS["text"])

    ax.set_ylabel("Annual Revenue (SAR Million)", fontweight="bold")
    ax.set_title("Revenue Projection by Pricing Model", fontsize=14, fontweight="bold",
                 color=COLORS["primary"], pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels)
    ax.set_ylim(0, max(PRICING_ENTERPRISE) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, frameon=True, edgecolor=COLORS["grid"], loc="upper left")
    plt.tight_layout()
    _save_chart(fig, "pricing_scenarios.png")
    print("  ✓ Pricing scenarios chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 6: 5-Year Revenue Forecast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_revenue_forecast() -> None:
    fc = DATA["revenue_forecast"]
    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    hospitals = [fc[f"year_{i}"]["hospitals"] for i in range(1, 6)]
    arr = [fc[f"year_{i}"]["arr_sar_m"] for i in range(1, 6)]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    bars = ax1.bar(year_labels, arr, color=COLORS["secondary"], width=0.5,
                   edgecolor="white", zorder=3, alpha=0.9)
    bars[-1].set_color(COLORS["accent"])

    for bar, val in zip(bars, arr):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"SAR {val}M", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color=COLORS["text"])

    ax1.set_ylabel("Annual Recurring Revenue (SAR Million)", fontweight="bold",
                   color=COLORS["secondary"])
    ax1.set_ylim(0, max(arr) * 1.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(year_labels, hospitals, "o-", color=COLORS["accent"],
             linewidth=2.5, markersize=8, zorder=4)
    for i, h in enumerate(hospitals):
        ax2.text(i, h + 2.5, str(h), ha="center", fontsize=10, fontweight="bold",
                 color=COLORS["accent"])
    ax2.set_ylabel("Active Customers", fontweight="bold", color=COLORS["accent"])
    ax2.set_ylim(0, max(hospitals) * 1.3)
    ax2.spines["top"].set_visible(False)

    ax1.set_title("5-Year Revenue Forecast & Adoption", fontsize=14,
                  fontweight="bold", color=COLORS["primary"], pad=15)

    growth_labels = ["—", "220%", "144%", "78%", "39%"]
    for i, g in enumerate(growth_labels):
        if g != "—":
            ax1.text(i, -1.8, f"+{g}", ha="center", fontsize=9,
                     color=COLORS["success"], fontweight="bold")

    plt.tight_layout()
    _save_chart(fig, "revenue_forecast.png")
    print("  ✓ Revenue forecast chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 7: Unit Economics Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_unit_economics() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    metrics = [
        ("CAC", "SAR 180K", "Cost to acquire\none customer", COLORS["accent"]),
        ("ACV", "SAR 900K", "Average contract\nvalue per year", COLORS["secondary"]),
        ("LTV", "SAR 3.6M", "Lifetime value\n(4-year avg.)", COLORS["success"]),
        ("LTV:CAC", "20x", "Strong unit\neconomics", COLORS["primary"]),
        ("Gross Margin", "80%", "Software\nmargins", COLORS["success"]),
        ("NRR", "125%", "Net revenue\nretention", COLORS["accent2"]),
    ]

    for ax, (name, value, desc, color) in zip(axes.flat, metrics):
        ax.text(0.5, 0.65, value, ha="center", va="center", fontsize=26,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.35, name, ha="center", va="center", fontsize=14,
                fontweight="bold", color=COLORS["text"], transform=ax.transAxes)
        ax.text(0.5, 0.15, desc, ha="center", va="center", fontsize=9,
                color=COLORS["muted"], transform=ax.transAxes)
        ax.add_patch(FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9, transform=ax.transAxes,
            boxstyle="round,pad=0.02", facecolor=COLORS["light"],
            edgecolor=color, linewidth=2, alpha=0.3))
        ax.axis("off")

    fig.suptitle("Unit Economics Summary", fontsize=16, fontweight="bold",
                 color=COLORS["primary"], y=1.02)
    plt.tight_layout()
    _save_chart(fig, "unit_economics.png")
    print("  ✓ Unit economics chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHART_GENERATORS = [
    ("TAM/SAM/SOM",           generate_tam_sam_som),
    ("Market Growth",         generate_market_growth),
    ("Hospital Segmentation", generate_hospital_segmentation),
    ("Regional Distribution", generate_regional_distribution),
    ("Pricing Scenarios",     generate_pricing_scenarios),
    ("Revenue Forecast",      generate_revenue_forecast),
    ("Unit Economics",        generate_unit_economics),
]


if __name__ == "__main__":
    print("\n🏥 Saudi Healthtech Market Analysis — Generating Charts\n")
    print(f"Output directory: {OUTPUT_DIR}\n")
    for _, gen in CHART_GENERATORS:
        gen()
    chart_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(f".{CHART_FORMAT}")])
    print(f"\n✅ Generated {chart_count} charts in {OUTPUT_DIR}/\n")
