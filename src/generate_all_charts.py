"""
Saudi Clinical Data Infrastructure — Market Analysis Chart Generation
=====================================================================

Generates all 9 visualizations for the clinical data infrastructure
market sizing & entry strategy deliverable.

Usage:
    python src/generate_all_charts.py

Output:
    9 PNG files in the output/ directory at 200 DPI.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration & Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

COMPETITOR_COLORS: dict[str, str] = {
    "LynxCare":         "#4A90D9",
    "Savana":           "#7B7B7B",
    "IOMED":            "#2D9A4E",
    "Mendel":           "#9B59B6",
    "IQVIA":            "#E74C3C",
    "MedFlow (Target)": "#E8792B",
}

REGION_COLORS: list[str] = [
    COLORS["primary"], COLORS["secondary"], "#5B8AC4",
    COLORS["accent2"], COLORS["muted"],
]

CHART_DPI: int = 200
CHART_FORMAT: str = "png"

HEALTHCARE_BASE_VALUE_SAR_B: float = 67.2
HEALTHCARE_CAGR: float = 0.067
PROJECTION_START_YEAR: int = 2023
PROJECTION_END_YEAR: int = 2030

TIER_LABELS: list[str] = [
    "Tier 1\n(Large Groups)", "Tier 2\n(Regional Chains)",
    "Tier 3\n(Single Site)", "Tier 4\n(Small/Specialty)",
]
TIER_COUNTS: list[int] = [15, 30, 45, 35]
TIER_AVG_BEDS: list[int] = [300, 150, 100, 35]
TIER_AVG_REVENUE_SAR_M: list[int] = [800, 250, 150, 60]

# Pricing model projections (SAR M, Years 1–5) — Annual hospital contracts
PRICING_ANNUAL_CONTRACT: list[float] = [5.6, 19.6, 48.0, 81.6, 120.0]
PRICING_PER_SPECIALTY: list[float] = [3.2, 10.8, 25.2, 42.0, 58.0]
PRICING_ENTERPRISE: list[float] = [8.0, 28.0, 62.0, 105.0, 155.0]

# Radar chart capability dimensions
RADAR_CATEGORIES: list[str] = [
    "Clinical NLP", "OMOP\nExpertise", "Regional\nPresence",
    "FHIR\nIntegration", "Pricing", "Scalability",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matplotlib Global Style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT_DIR: str = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR: str = os.path.join(ROOT_DIR, "output")
DATA_DIR: str = os.path.join(ROOT_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_market_data() -> dict[str, Any]:
    data_path = os.path.join(DATA_DIR, "market_data.json")
    with open(data_path) as f:
        return json.load(f)


DATA: dict[str, Any] = load_market_data()


def _save_chart(fig: plt.Figure, filename: str) -> None:
    fig.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=CHART_DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 1: TAM / SAM / SOM — Concentric Circle Diagram
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
        circle = plt.Circle((0.5, 0.45), radius, color=color, alpha=0.85 - i * 0.1)
        ax.add_patch(circle)

    ax.text(0.5, 0.45, f"SOM\nSAR {sizing['som_year5_sar_m']}M\n(Year 5 Target)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(0.5, 0.78, f"SAM — SAR {sizing['sam_sar_m']/1000:.1f}B",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(0.5, 0.05, f"TAM — SAR {sizing['tam_sar_b']}B",
            ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["primary"])

    descriptions: list[tuple[str, str, float]] = [
        ("TAM", "Clinical data infrastructure\nspend across KSA healthcare", 0.88),
        ("SAM", "Private hospitals with\nFHIR-enabled EMR systems", 0.72),
        ("SOM", "Realistically capturable\nin 5-year horizon", 0.45),
    ]
    for _, desc, y in descriptions:
        ax.annotate(desc, xy=(0.82, y), fontsize=9, color=COLORS["muted"],
                    ha="left", va="center")

    ax.set_xlim(-0.1, 1.4)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("Market Sizing: TAM / SAM / SOM", fontsize=16, fontweight="bold",
                 color=COLORS["primary"], y=0.97)
    ax.text(0.5, 0.98, "Clinical Data Infrastructure — Saudi Arabia",
            ha="center", va="top", fontsize=11, color=COLORS["muted"],
            transform=ax.transAxes)

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
            "Source: Ministry of Health Statistical Yearbook, Frost & Sullivan MENA Health IT Report (2024)",
            transform=ax.transAxes, fontsize=8, color=COLORS["muted"])

    plt.tight_layout()
    _save_chart(fig, "market_growth.png")
    print("  ✓ Market growth chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 3: Hospital Data Readiness Assessment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_data_maturity() -> None:
    maturity = DATA["data_maturity"]

    categories = [
        "Cannot query\nown clinical data",
        "FHIR endpoint\nactive",
        "Structured\nregistries",
        "OMOP CDM\nadopted",
    ]
    values = [
        maturity["hospitals_cannot_query_own_data_pct"],
        maturity["hospitals_with_fhir_endpoint_pct"],
        maturity["hospitals_with_structured_registries_pct"],
        maturity["hospitals_with_omop_pct"],
    ]
    bar_colors = ["#E8792B", "#2E5090", "#2D9A4E", "#1B2A4A"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bar chart — data readiness gap
    bars = ax1.barh(categories, values, color=bar_colors, height=0.55, edgecolor="white", zorder=3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 f"{val}%", va="center", fontsize=13, fontweight="bold",
                 color=COLORS["text"])

    ax1.set_xlabel("Percentage of Private Hospitals", fontweight="bold")
    ax1.set_title("Data Readiness Gap", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax1.set_xlim(0, 105)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Stacked bar showing opportunity
    opportunity_labels = ["Queryable\n(OMOP)", "Structured but\nnot queryable", "Unstructured\n(opportunity)"]
    opportunity_vals = [
        maturity["hospitals_with_omop_pct"],
        maturity["hospitals_with_structured_registries_pct"] - maturity["hospitals_with_omop_pct"],
        100 - maturity["hospitals_with_structured_registries_pct"],
    ]
    opportunity_colors = ["#1B2A4A", "#2D9A4E", "#E8792B"]

    bottom = 0
    for val, color, label in zip(opportunity_vals, opportunity_colors, opportunity_labels):
        ax2.bar(0, val, bottom=bottom, color=color, width=0.5, edgecolor="white",
                label=label, zorder=3)
        if val > 5:
            ax2.text(0, bottom + val / 2, f"{val}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")
        bottom += val

    ax2.set_ylabel("Percentage of Hospitals", fontweight="bold")
    ax2.set_title("Clinical Data Structuring Opportunity", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax2.set_ylim(0, 105)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.legend(fontsize=10, loc="upper right", frameon=True, edgecolor=COLORS["grid"])

    fig.suptitle("Hospital Clinical Data Maturity — Saudi Arabia", fontsize=15,
                 fontweight="bold", color=COLORS["primary"], y=1.02)

    plt.tight_layout()
    _save_chart(fig, "data_maturity.png")
    print("  ✓ Data maturity chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 4: Competitive Capability Radar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_competitive_radar() -> None:
    competitors: dict[str, dict[str, int]] = DATA["competitors"]
    n_categories = len(RADAR_CATEGORIES)

    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    capability_keys = [
        "clinical_nlp", "omop_expertise", "regional_presence",
        "fhir_integration", "pricing_competitiveness", "scalability",
    ]

    for name, scores in competitors.items():
        values = [scores[key] for key in capability_keys]
        values += values[:1]

        is_target = name == "MedFlow (Target)"
        line_width = 3 if is_target else 1.5
        fill_alpha = 0.18 if is_target else 0.05

        ax.plot(angles, values, "o-", linewidth=line_width, label=name,
                color=COMPETITOR_COLORS[name])
        ax.fill(angles, values, alpha=fill_alpha, color=COMPETITOR_COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_CATEGORIES, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color=COLORS["muted"])
    ax.spines["polar"].set_color(COLORS["grid"])

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10,
              frameon=True, fancybox=True, shadow=False, edgecolor=COLORS["grid"])

    ax.set_title("Competitive Capability Assessment", fontsize=15, fontweight="bold",
                 color=COLORS["primary"], pad=30)

    _save_chart(fig, "competitive_radar.png")
    print("  ✓ Competitive radar chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 5: Hospital Segmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_hospital_segmentation() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Hospital count by tier
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

    # Right: Revenue opportunity
    bubble_scale_factor = 5
    x_pos = list(range(len(TIER_LABELS)))
    ax2.scatter(x_pos, TIER_AVG_REVENUE_SAR_M,
                s=[b * bubble_scale_factor for b in TIER_AVG_BEDS],
                c=TIER_COLORS, alpha=0.8, edgecolors="white", linewidth=2, zorder=3)

    for i, (rev, beds) in enumerate(zip(TIER_AVG_REVENUE_SAR_M, TIER_AVG_BEDS)):
        ax2.text(i, rev + 30, f"SAR {rev}M\n({beds} beds avg)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=COLORS["text"])

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(TIER_LABELS, fontsize=9)
    ax2.set_ylabel("Avg. Annual Revenue (SAR M)", fontweight="bold")
    ax2.set_title("Revenue & Scale by Tier", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax2.set_ylim(0, max(TIER_AVG_REVENUE_SAR_M) * 1.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    total_hospitals = sum(TIER_COUNTS)
    fig.suptitle(f"Private Hospital Segmentation — Saudi Arabia ({total_hospitals} Hospitals)",
                 fontsize=15, fontweight="bold", color=COLORS["primary"], y=1.02)

    plt.tight_layout()
    _save_chart(fig, "hospital_segmentation.png")
    print("  ✓ Hospital segmentation chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 6: Regional Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_regional_distribution() -> None:
    regional_data = DATA["regional_distribution"]
    regions = list(regional_data.keys())
    hospitals = [regional_data[r]["hospitals"] for r in regions]
    pcts = [regional_data[r]["pct"] for r in regions]

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
        t.set_fontsize(10)
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(9)

    ax2.set_title("Market Concentration", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)

    total = sum(hospitals)
    ax2.text(0, 0, f"{total}\nHospitals", ha="center", va="center", fontsize=14,
             fontweight="bold", color=COLORS["primary"])

    fig.suptitle("Regional Distribution of Private Hospitals", fontsize=15,
                 fontweight="bold", color=COLORS["primary"], y=1.02)

    plt.tight_layout()
    _save_chart(fig, "regional_distribution.png")
    print("  ✓ Regional distribution chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 7: Pricing Model Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_pricing_scenarios() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    x = np.arange(len(year_labels))
    width = 0.25

    models = [
        (PRICING_PER_SPECIALTY, "Per-Specialty PoC (SAR 50-100K/dept)"),
        (PRICING_ANNUAL_CONTRACT, "Annual Hospital Contract (SAR 500K-1.5M)"),
        (PRICING_ENTERPRISE, "Enterprise License (SAR 2-5M/yr)"),
    ]

    for i, (data, label) in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data, width, label=label,
                      color=SCENARIO_COLORS[i], edgecolor="white", zorder=3)
        last_bar = bars[-1]
        ax.text(last_bar.get_x() + last_bar.get_width() / 2,
                last_bar.get_height() + 1,
                f"SAR {data[-1]}M", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=COLORS["text"])

    ax.set_ylabel("Annual Revenue (SAR Million)", fontweight="bold")
    ax.set_title("Revenue Projection by Pricing Model", fontsize=14, fontweight="bold",
                 color=COLORS["primary"], pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels)
    ax.set_ylim(0, max(PRICING_ENTERPRISE) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, frameon=True, edgecolor=COLORS["grid"], loc="upper left")

    ax.annotate("Recommended: Annual\nHospital Contract",
                xy=(4, 120), fontsize=10, fontweight="bold", color=COLORS["accent"],
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E8",
                          edgecolor=COLORS["accent"]))

    plt.tight_layout()
    _save_chart(fig, "pricing_scenarios.png")
    print("  ✓ Pricing scenarios chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 8: 5-Year Revenue Forecast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_revenue_forecast() -> None:
    forecast = DATA["revenue_forecast"]
    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    hospitals = [forecast[f"year_{i}"]["hospitals"] for i in range(1, 6)]
    arr = [forecast[f"year_{i}"]["arr_sar_m"] for i in range(1, 6)]

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

    ax2.set_ylabel("Active Hospitals", fontweight="bold", color=COLORS["accent"])
    ax2.set_ylim(0, max(hospitals) * 1.3)
    ax2.spines["top"].set_visible(False)

    ax1.set_title("5-Year Revenue Forecast & Hospital Adoption", fontsize=14,
                  fontweight="bold", color=COLORS["primary"], pad=15)

    # YoY growth rates
    growth_labels = ["—", "250%", "145%", "70%", "47%"]
    for i, g in enumerate(growth_labels):
        if g != "—":
            ax1.text(i, -5, f"+{g}", ha="center", fontsize=9,
                     color=COLORS["success"], fontweight="bold")

    plt.tight_layout()
    _save_chart(fig, "revenue_forecast.png")
    print("  ✓ Revenue forecast chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 9: Unit Economics Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_unit_economics() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    metrics: list[tuple[str, str, float, str, str]] = [
        ("CAC", "SAR 200K", 200, "Cost to acquire\none hospital", COLORS["accent"]),
        ("ACV", "SAR 1.4M", 1400, "Average contract\nvalue per year", COLORS["secondary"]),
        ("LTV", "SAR 3.7M", 3700, "Lifetime value\n(3-year avg.)", COLORS["success"]),
        ("LTV:CAC", "18.5x", 18.5, "Strong unit\neconomics", COLORS["primary"]),
        ("Gross Margin", "78%", 78, "Infrastructure\nmargins", COLORS["success"]),
        ("NRR", "130%", 130, "Net revenue\nretention", COLORS["accent2"]),
    ]

    for ax, (name, value, _, desc, color) in zip(axes.flat, metrics):
        ax.text(0.5, 0.65, value, ha="center", va="center", fontsize=26,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.35, name, ha="center", va="center", fontsize=14,
                fontweight="bold", color=COLORS["text"], transform=ax.transAxes)
        ax.text(0.5, 0.15, desc, ha="center", va="center", fontsize=9,
                color=COLORS["muted"], transform=ax.transAxes)

        rect = FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9, transform=ax.transAxes,
            boxstyle="round,pad=0.02", facecolor=COLORS["light"],
            edgecolor=color, linewidth=2, alpha=0.3,
        )
        ax.add_patch(rect)
        ax.axis("off")

    fig.suptitle("Unit Economics Summary", fontsize=16, fontweight="bold",
                 color=COLORS["primary"], y=1.02)

    plt.tight_layout()
    _save_chart(fig, "unit_economics.png")
    print("  ✓ Unit economics chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHART_GENERATORS: list[tuple[str, callable]] = [
    ("TAM/SAM/SOM",           generate_tam_sam_som),
    ("Market Growth",         generate_market_growth),
    ("Data Maturity",         generate_data_maturity),
    ("Competitive Radar",     generate_competitive_radar),
    ("Hospital Segmentation", generate_hospital_segmentation),
    ("Regional Distribution", generate_regional_distribution),
    ("Pricing Scenarios",     generate_pricing_scenarios),
    ("Revenue Forecast",      generate_revenue_forecast),
    ("Unit Economics",        generate_unit_economics),
]


if __name__ == "__main__":
    print("\n🏥 Saudi Clinical Data Infrastructure — Generating Charts\n")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for _, generator in CHART_GENERATORS:
        generator()

    chart_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(f".{CHART_FORMAT}")])
    print(f"\n✅ All charts generated successfully in {OUTPUT_DIR}/")
    print(f"   Files: {chart_count} visualizations\n")
