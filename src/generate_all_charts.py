"""
Saudi Healthtech Market Analysis — Chart Generation
====================================================

Generates all 9 visualizations for the market sizing & entry strategy
deliverable. Charts follow a McKinsey/BCG-inspired design language for
a consulting-grade portfolio presentation.

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

# McKinsey/BCG-inspired color palette
COLORS: dict[str, str] = {
    "primary":   "#1B2A4A",   # Deep navy
    "secondary": "#2E5090",   # Royal blue
    "accent":    "#E8792B",   # Warm orange
    "accent2":   "#D4A843",   # Gold
    "light":     "#F0F4F8",   # Light gray-blue
    "success":   "#2D9A4E",   # Forest green
    "text":      "#1D2530",   # Dark text
    "muted":     "#7A8B9C",   # Muted text
    "bg":        "#FFFFFF",   # White background
    "grid":      "#E8ECF0",   # Light grid lines
}

# Hospital tier colors (Tier 1 → 4, dark → light)
TIER_COLORS: list[str] = ["#1B2A4A", "#2E5090", "#5B8AC4", "#A3C4E8"]

# Scenario colors: Conservative (green), Base (blue), Aggressive (orange)
SCENARIO_COLORS: list[str] = ["#2D9A4E", "#2E5090", "#E8792B"]

# Competitor colors for radar chart
COMPETITOR_COLORS: dict[str, str] = {
    "Nuance/Microsoft": "#4A90D9",
    "3M (Solventum)":   "#7B7B7B",
    "Sahl AI":          "#2D9A4E",
    "Nuxera":           "#9B59B6",
    "Nym Health":       "#E74C3C",
    "MedFlow (Target)": "#E8792B",
}

# Region colors
REGION_COLORS: list[str] = [
    COLORS["primary"], COLORS["secondary"], "#5B8AC4",
    COLORS["accent2"], COLORS["muted"],
]

# Chart output settings
CHART_DPI: int = 200
CHART_FORMAT: str = "png"

# Market assumptions
HEALTHCARE_BASE_VALUE_SAR_B: float = 67.2
HEALTHCARE_CAGR: float = 0.067
PROJECTION_START_YEAR: int = 2023
PROJECTION_END_YEAR: int = 2030
DENIAL_REDUCTION_TARGET_PCT: float = 0.35

# Hospital segmentation constants
TIER_LABELS: list[str] = [
    "Tier 1\n(Large Groups)", "Tier 2\n(Regional Chains)",
    "Tier 3\n(Single Site)", "Tier 4\n(Small/Specialty)",
]
TIER_COUNTS: list[int] = [15, 30, 45, 35]
TIER_AVG_BEDS: list[int] = [300, 150, 100, 35]
TIER_AVG_REVENUE_SAR_M: list[int] = [800, 250, 150, 60]

# Pricing model projections (SAR M, Years 1–5)
PRICING_PER_ENCOUNTER: list[float] = [3.2, 12.1, 26.4, 43.8, 58.0]
PRICING_PER_BED: list[float] = [2.8, 9.5, 20.1, 33.2, 43.0]
PRICING_ENTERPRISE: list[float] = [5.5, 16.8, 34.2, 55.0, 72.0]

# Denial economics scenarios
DENIAL_SCENARIOS: list[str] = ["Conservative", "Base Case", "Aggressive"]
DENIAL_RATES_PCT: list[int] = [15, 20, 25]
COSTS_PER_DENIED_CLAIM_SAR: list[int] = [150, 225, 300]
CLAIMS_PER_HOSPITAL_YEAR: list[int] = [45_000, 60_000, 80_000]

# Radar chart capability dimensions
RADAR_CATEGORIES: list[str] = [
    "Arabic NLP", "NPHIES\nIntegration", "AI Depth",
    "Local\nPresence", "Pricing", "Scalability",
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
    """Load the market data JSON that drives all data-driven charts.

    Returns:
        Dictionary containing market sizing, competitor scores,
        regional distribution, and revenue forecast data.
    """
    data_path = os.path.join(DATA_DIR, "market_data.json")
    with open(data_path) as f:
        return json.load(f)


DATA: dict[str, Any] = load_market_data()


def _save_chart(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the output directory.

    Args:
        fig: The matplotlib Figure to save.
        filename: Output filename (without directory path).
    """
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
    """Generate the TAM/SAM/SOM concentric circle market sizing chart.

    Visualizes the three-layer market sizing model:
    - TAM (SAR 2.94B): Total addressable market for clinical documentation
    - SAM (SAR 734M): Serviceable segment — private hospitals with >50 beds
    - SOM (SAR 73.4M): Obtainable market within 5-year horizon
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    labels_data = [
        ("TAM\nSAR 2.94B", "#D6E4F0"),
        ("SAM\nSAR 734M", "#5B8AC4"),
        ("SOM\nSAR 73.4M", "#1B2A4A"),
    ]

    # Draw concentric circles (largest first)
    for i, (_, color) in enumerate(labels_data):
        radius = 0.3 + (2 - i) * 0.25
        circle = plt.Circle((0.5, 0.45), radius, color=color, alpha=0.85 - i * 0.1)
        ax.add_patch(circle)

    # Center label (SOM)
    ax.text(0.5, 0.45, "SOM\nSAR 73.4M\n(Year 5 Target)", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.text(0.5, 0.78, "SAM — SAR 734M", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(0.5, 0.05, "TAM — SAR 2.94B", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLORS["primary"])

    # Descriptions on right side
    descriptions: list[tuple[str, str, float]] = [
        ("TAM", "All clinical documentation &\nclaims processing spend in KSA", 0.88),
        ("SAM", "AI documentation tools for\nprivate hospitals with >50 beds", 0.72),
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
    ax.text(0.5, 0.98, "AI Clinical Documentation Tools — Saudi Arabia",
            ha="center", va="top", fontsize=11, color=COLORS["muted"],
            transform=ax.transAxes)

    plt.tight_layout()
    _save_chart(fig, "tam_sam_som.png")
    print("  ✓ TAM/SAM/SOM chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 2: Healthcare Market Growth Projection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_market_growth() -> None:
    """Generate the Saudi healthcare market growth bar chart (2023–2030).

    Projects total healthcare expenditure using the 6.7% CAGR from
    MOH/Frost & Sullivan data, highlighting the 2030 Vision target.
    """
    years = list(range(PROJECTION_START_YEAR, PROJECTION_END_YEAR + 1))
    values = [HEALTHCARE_BASE_VALUE_SAR_B * (1 + HEALTHCARE_CAGR) ** (y - PROJECTION_START_YEAR)
              for y in years]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(years, values, color=COLORS["secondary"], width=0.6,
                  edgecolor="white", linewidth=0.5, zorder=3)
    bars[-1].set_color(COLORS["accent"])  # Highlight 2030

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
# Chart 3: Claims Denial Economics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_denial_economics() -> None:
    """Generate the claims denial economics dual-panel chart.

    Left panel: Annual denial cost per hospital across three scenarios.
    Right panel: Revenue recovery potential with 35% denial rate reduction.
    Demonstrates the core value proposition for AI documentation tools.
    """
    annual_denial_cost = [
        (rate / 100) * cost * claims / 1e6
        for rate, cost, claims in zip(DENIAL_RATES_PCT, COSTS_PER_DENIED_CLAIM_SAR, CLAIMS_PER_HOSPITAL_YEAR)
    ]
    recovery = [x * DENIAL_REDUCTION_TARGET_PCT for x in annual_denial_cost]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Annual denial cost
    bars1 = ax1.bar(DENIAL_SCENARIOS, annual_denial_cost, color=SCENARIO_COLORS,
                    width=0.5, edgecolor="white", zorder=3)
    for bar, val in zip(bars1, annual_denial_cost):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                 f"SAR {val:.2f}M", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color=COLORS["text"])

    ax1.set_ylabel("Annual Cost (SAR Million)", fontweight="bold")
    ax1.set_title("Annual Denial Cost per Hospital", fontsize=13, fontweight="bold",
                  color=COLORS["primary"], pad=10)
    ax1.set_ylim(0, max(annual_denial_cost) * 1.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Recovery potential
    bars2 = ax2.bar(DENIAL_SCENARIOS, recovery, color=SCENARIO_COLORS,
                    width=0.5, edgecolor="white", zorder=3)
    for bar, val in zip(bars2, recovery):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f"SAR {val:.2f}M", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color=COLORS["text"])

    ax2.set_ylabel("Annual Recovery (SAR Million)", fontweight="bold")
    ax2.set_title(f"Revenue Recovery with {int(DENIAL_REDUCTION_TARGET_PCT * 100)}% Denial Reduction",
                  fontsize=13, fontweight="bold", color=COLORS["primary"], pad=10)
    ax2.set_ylim(0, max(recovery) * 1.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Claims Denial Economics — Per Hospital Analysis", fontsize=15,
                 fontweight="bold", color=COLORS["primary"], y=1.02)

    plt.tight_layout()
    _save_chart(fig, "denial_economics.png")
    print("  ✓ Denial economics chart")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 4: Competitive Capability Radar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_competitive_radar() -> None:
    """Generate the competitive capability radar chart.

    Plots 6 capability dimensions for all market participants including
    the target positioning (MedFlow). Highlights the strategic gap in
    Arabic NLP + NPHIES integration that no incumbent fills.
    """
    competitors: dict[str, dict[str, int]] = DATA["competitors"]
    n_categories = len(RADAR_CATEGORIES)

    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    capability_keys = [
        "arabic_nlp", "nphies_integration", "ai_depth",
        "local_presence", "pricing_competitiveness", "scalability",
    ]

    for name, scores in competitors.items():
        values = [scores[key] for key in capability_keys]
        values += values[:1]

        is_target = name == "MedFlow (Target)"
        line_width = 3 if is_target else 1.5
        fill_alpha = 0.15 if is_target else 0.05

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
    """Generate the hospital segmentation dual-panel chart.

    Left panel: Horizontal bar chart of hospital counts by tier.
    Right panel: Bubble scatter showing revenue and scale by tier,
    with bubble size proportional to average bed count.
    """
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

    # Right: Revenue opportunity (bubble scatter)
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
    """Generate the regional distribution dual-panel chart.

    Left panel: Bar chart of hospital counts by region.
    Right panel: Donut chart showing market concentration across
    Saudi Arabia's five major healthcare regions.
    """
    regional_data = DATA["regional_distribution"]
    regions = list(regional_data.keys())
    hospitals = [regional_data[r]["hospitals"] for r in regions]
    pcts = [regional_data[r]["pct"] for r in regions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Bar chart
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

    # Donut chart
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
    """Generate the pricing scenario comparison grouped bar chart.

    Compares three pricing models (per-encounter, per-bed, enterprise
    license) across the 5-year forecast horizon, with Year 5 revenue
    labels and a recommendation callout for the hybrid approach.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    x = np.arange(len(year_labels))
    width = 0.25

    models = [
        (PRICING_PER_ENCOUNTER, "Per Encounter (SAR 8-15/enc)"),
        (PRICING_PER_BED, "Per Bed (SAR 500-1,200/bed/mo)"),
        (PRICING_ENTERPRISE, "Enterprise License (SAR 1.5-4M/yr)"),
    ]

    for i, (data, label) in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data, width, label=label,
                      color=SCENARIO_COLORS[i], edgecolor="white", zorder=3)
        # Label Year 5 value
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

    ax.annotate("Recommended: Hybrid\n(Per Encounter + Per Bed)",
                xy=(4, 58), fontsize=10, fontweight="bold", color=COLORS["accent"],
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
    """Generate the revenue forecast bar + line combo chart.

    Combines ARR bars (left axis) with hospital adoption line (right axis)
    to show the dual growth trajectory. YoY growth rates annotated below bars.
    """
    forecast = DATA["revenue_forecast"]
    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    hospitals = [forecast[f"year_{i}"]["hospitals"] for i in range(1, 6)]
    arr = [forecast[f"year_{i}"]["arr_sar_m"] for i in range(1, 6)]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Revenue bars
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

    # Hospital adoption line (secondary axis)
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
    growth_labels = ["—", "252%", "113%", "65%", "41%"]
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
    """Generate the unit economics KPI dashboard card grid.

    Displays six key SaaS metrics in a 2×3 card layout:
    CAC, ACV, LTV, LTV:CAC ratio, gross margin, and net revenue retention.
    Each card includes the metric value, name, and contextual description.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    metrics: list[tuple[str, str, float, str, str]] = [
        ("CAC", "SAR 180K", 180, "Cost to acquire\none hospital", COLORS["accent"]),
        ("ACV", "SAR 1.02M", 1020, "Average contract\nvalue per year", COLORS["secondary"]),
        ("LTV", "SAR 4.1M", 4100, "Lifetime value\n(4-year avg.)", COLORS["success"]),
        ("LTV:CAC", "22.8x", 22.8, "Strong unit\neconomics", COLORS["primary"]),
        ("Gross Margin", "82%", 82, "Software-like\nmargins", COLORS["success"]),
        ("NRR", "135%", 135, "Net revenue\nretention", COLORS["accent2"]),
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
    ("Denial Economics",      generate_denial_economics),
    ("Competitive Radar",     generate_competitive_radar),
    ("Hospital Segmentation", generate_hospital_segmentation),
    ("Regional Distribution", generate_regional_distribution),
    ("Pricing Scenarios",     generate_pricing_scenarios),
    ("Revenue Forecast",      generate_revenue_forecast),
    ("Unit Economics",        generate_unit_economics),
]


if __name__ == "__main__":
    print("\n🏥 Saudi Healthtech Market Analysis — Generating Charts\n")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for _, generator in CHART_GENERATORS:
        generator()

    chart_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(f".{CHART_FORMAT}")])
    print(f"\n✅ All charts generated successfully in {OUTPUT_DIR}/")
    print(f"   Files: {chart_count} visualizations\n")
