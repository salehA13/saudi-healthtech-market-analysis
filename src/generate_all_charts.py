"""
Saudi Healthtech Market Analysis — Chart Generation
Generates all visualizations for the market sizing & entry strategy deliverable.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style Configuration ──────────────────────────────────────────────────────
# McKinsey/BCG-inspired palette
COLORS = {
    'primary': '#1B2A4A',      # Deep navy
    'secondary': '#2E5090',    # Royal blue
    'accent': '#E8792B',       # Warm orange
    'accent2': '#D4A843',      # Gold
    'light': '#F0F4F8',        # Light gray-blue
    'success': '#2D9A4E',      # Forest green
    'text': '#1D2530',         # Dark text
    'muted': '#7A8B9C',        # Muted text
    'bg': '#FFFFFF',           # White bg
    'grid': '#E8ECF0',        # Light grid
}

TIER_COLORS = ['#1B2A4A', '#2E5090', '#5B8AC4', '#A3C4E8']
SCENARIO_COLORS = ['#2D9A4E', '#2E5090', '#E8792B']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': COLORS['grid'],
    'axes.grid': True,
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.6,
    'grid.linewidth': 0.5,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.json')
    with open(data_path) as f:
        return json.load(f)

DATA = load_data()


# ── Chart 1: TAM / SAM / SOM ────────────────────────────────────────────────
def generate_tam_sam_som():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sizes = [2940, 734, 73.4]
    labels = ['TAM\nSAR 2.94B', 'SAM\nSAR 734M', 'SOM\nSAR 73.4M']
    colors = ['#D6E4F0', '#5B8AC4', '#1B2A4A']
    
    # Concentric circles
    for i, (size, label, color) in enumerate(zip(sizes, labels, colors)):
        radius = 0.3 + (2 - i) * 0.25
        circle = plt.Circle((0.5, 0.45), radius, color=color, alpha=0.85 - i*0.1)
        ax.add_patch(circle)
    
    # Labels
    ax.text(0.5, 0.45, 'SOM\nSAR 73.4M\n(Year 5 Target)', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    ax.text(0.5, 0.78, 'SAM — SAR 734M', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(0.5, 0.05, 'TAM — SAR 2.94B', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    
    # Descriptions on right side
    descs = [
        ('TAM', 'All clinical documentation &\nclaims processing spend in KSA', 0.88),
        ('SAM', 'AI documentation tools for\nprivate hospitals with >50 beds', 0.72),
        ('SOM', 'Realistically capturable\nin 5-year horizon', 0.45),
    ]
    for name, desc, y in descs:
        ax.annotate(desc, xy=(0.82, y), fontsize=9, color=COLORS['muted'],
                   ha='left', va='center')
    
    ax.set_xlim(-0.1, 1.4)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    
    fig.suptitle('Market Sizing: TAM / SAM / SOM', fontsize=16, fontweight='bold',
                color=COLORS['primary'], y=0.97)
    ax.text(0.5, 0.98, 'AI Clinical Documentation Tools — Saudi Arabia',
           ha='center', va='top', fontsize=11, color=COLORS['muted'],
           transform=ax.transAxes)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'tam_sam_som.png'), dpi=200, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ TAM/SAM/SOM chart")


# ── Chart 2: Healthcare Market Growth ────────────────────────────────────────
def generate_market_growth():
    years = list(range(2023, 2031))
    base = 67.2
    cagr = 0.067
    values = [base * (1 + cagr) ** (y - 2023) for y in years]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(years, values, color=COLORS['secondary'], width=0.6, 
                  edgecolor='white', linewidth=0.5, zorder=3)
    
    # Highlight 2030
    bars[-1].set_color(COLORS['accent'])
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'SAR {val:.1f}B', ha='center', va='bottom', fontsize=9,
               fontweight='bold', color=COLORS['text'])
    
    # CAGR annotation
    ax.annotate('6.7% CAGR', xy=(2026.5, 82), fontsize=14, fontweight='bold',
               color=COLORS['accent'], ha='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['light'], 
                        edgecolor=COLORS['accent'], alpha=0.9))
    
    ax.set_ylabel('Market Size (SAR Billion)', fontweight='bold')
    ax.set_title('Saudi Healthcare Market Projection (2023–2030)', 
                fontsize=14, fontweight='bold', color=COLORS['primary'], pad=15)
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(years)
    
    # Source
    ax.text(0.0, -0.12, 'Source: Ministry of Health Statistical Yearbook, Frost & Sullivan MENA Health IT Report (2024)',
           transform=ax.transAxes, fontsize=8, color=COLORS['muted'])
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'market_growth.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Market growth chart")


# ── Chart 3: Denial Economics ────────────────────────────────────────────────
def generate_denial_economics():
    scenarios = ['Conservative', 'Base Case', 'Aggressive']
    denial_rates = [15, 20, 25]
    costs_per_claim = [150, 225, 300]
    claims = [45000, 60000, 80000]
    
    annual_denial_cost = [d/100 * c * cl / 1e6 for d, c, cl in zip(denial_rates, costs_per_claim, claims)]
    recovery_35pct = [x * 0.35 for x in annual_denial_cost]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Annual denial cost
    bars1 = ax1.bar(scenarios, annual_denial_cost, color=SCENARIO_COLORS, width=0.5, 
                    edgecolor='white', zorder=3)
    for bar, val in zip(bars1, annual_denial_cost):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'SAR {val:.2f}M', ha='center', va='bottom', fontsize=11,
                fontweight='bold', color=COLORS['text'])
    
    ax1.set_ylabel('Annual Cost (SAR Million)', fontweight='bold')
    ax1.set_title('Annual Denial Cost per Hospital', fontsize=13, fontweight='bold',
                 color=COLORS['primary'], pad=10)
    ax1.set_ylim(0, max(annual_denial_cost) * 1.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Recovery potential
    bars2 = ax2.bar(scenarios, recovery_35pct, color=SCENARIO_COLORS, width=0.5,
                    edgecolor='white', zorder=3)
    for bar, val in zip(bars2, recovery_35pct):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'SAR {val:.2f}M', ha='center', va='bottom', fontsize=11,
                fontweight='bold', color=COLORS['text'])
    
    ax2.set_ylabel('Annual Recovery (SAR Million)', fontweight='bold')
    ax2.set_title('Revenue Recovery with 35% Denial Reduction', fontsize=13, fontweight='bold',
                 color=COLORS['primary'], pad=10)
    ax2.set_ylim(0, max(recovery_35pct) * 1.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Claims Denial Economics — Per Hospital Analysis', fontsize=15, 
                fontweight='bold', color=COLORS['primary'], y=1.02)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'denial_economics.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Denial economics chart")


# ── Chart 4: Competitive Radar ──────────────────────────────────────────────
def generate_competitive_radar():
    competitors = DATA['competitors']
    categories = ['Arabic NLP', 'NPHIES\nIntegration', 'AI Depth', 
                  'Local\nPresence', 'Pricing', 'Scalability']
    N = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    comp_colors = {
        'Nuance/Microsoft': '#4A90D9',
        '3M (Solventum)': '#7B7B7B',
        'Sahl AI': '#2D9A4E',
        'Nuxera': '#9B59B6',
        'Nym Health': '#E74C3C',
        'MedFlow (Target)': '#E8792B',
    }
    
    for name, scores in competitors.items():
        values = [scores['arabic_nlp'], scores['nphies_integration'], scores['ai_depth'],
                 scores['local_presence'], scores['pricing_competitiveness'], scores['scalability']]
        values += values[:1]
        
        lw = 3 if name == 'MedFlow (Target)' else 1.5
        alpha = 0.15 if name == 'MedFlow (Target)' else 0.05
        
        ax.plot(angles, values, 'o-', linewidth=lw, label=name, color=comp_colors[name])
        ax.fill(angles, values, alpha=alpha, color=comp_colors[name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8, color=COLORS['muted'])
    ax.spines['polar'].set_color(COLORS['grid'])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=True,
             fancybox=True, shadow=False, edgecolor=COLORS['grid'])
    
    ax.set_title('Competitive Capability Assessment', fontsize=15, fontweight='bold',
                color=COLORS['primary'], pad=30)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'competitive_radar.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Competitive radar chart")


# ── Chart 5: Hospital Segmentation ──────────────────────────────────────────
def generate_hospital_segmentation():
    tiers = ['Tier 1\n(Large Groups)', 'Tier 2\n(Regional Chains)', 
             'Tier 3\n(Single Site)', 'Tier 4\n(Small/Specialty)']
    counts = [15, 30, 45, 35]
    avg_beds = [300, 150, 100, 35]
    avg_revenue = [800, 250, 150, 60]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Hospital count by tier
    bars = ax1.barh(tiers, counts, color=TIER_COLORS, height=0.6, edgecolor='white')
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val} hospitals', va='center', fontsize=11, fontweight='bold',
                color=COLORS['text'])
    
    ax1.set_xlabel('Number of Hospitals', fontweight='bold')
    ax1.set_title('Hospital Count by Tier', fontsize=13, fontweight='bold',
                 color=COLORS['primary'], pad=10)
    ax1.set_xlim(0, max(counts) * 1.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.invert_yaxis()
    
    # Right: Revenue opportunity (bubble-like bar)
    x_pos = range(len(tiers))
    scatter = ax2.scatter(x_pos, avg_revenue, s=[b*5 for b in avg_beds], 
                         c=TIER_COLORS, alpha=0.8, edgecolors='white', linewidth=2, zorder=3)
    
    for i, (rev, beds) in enumerate(zip(avg_revenue, avg_beds)):
        ax2.text(i, rev + 30, f'SAR {rev}M\n({beds} beds avg)',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COLORS['text'])
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tiers, fontsize=9)
    ax2.set_ylabel('Avg. Annual Revenue (SAR M)', fontweight='bold')
    ax2.set_title('Revenue & Scale by Tier', fontsize=13, fontweight='bold',
                 color=COLORS['primary'], pad=10)
    ax2.set_ylim(0, max(avg_revenue) * 1.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Private Hospital Segmentation — Saudi Arabia (125 Hospitals)',
                fontsize=15, fontweight='bold', color=COLORS['primary'], y=1.02)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'hospital_segmentation.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Hospital segmentation chart")


# ── Chart 6: Regional Distribution ──────────────────────────────────────────
def generate_regional_distribution():
    regions = list(DATA['regional_distribution'].keys())
    hospitals = [DATA['regional_distribution'][r]['hospitals'] for r in regions]
    pcts = [DATA['regional_distribution'][r]['pct'] for r in regions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    
    colors = [COLORS['primary'], COLORS['secondary'], '#5B8AC4', COLORS['accent2'], COLORS['muted']]
    
    # Bar chart
    bars = ax1.bar(regions, hospitals, color=colors, width=0.6, edgecolor='white', zorder=3)
    for bar, val, pct in zip(bars, hospitals, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val}\n({pct}%)', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=COLORS['text'])
    
    ax1.set_ylabel('Number of Private Hospitals', fontweight='bold')
    ax1.set_title('Hospitals by Region', fontsize=13, fontweight='bold',
                 color=COLORS['primary'], pad=10)
    ax1.set_ylim(0, max(hospitals) * 1.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Donut chart
    wedges, texts, autotexts = ax2.pie(hospitals, labels=regions, colors=colors,
                                        autopct='%1.0f%%', startangle=90,
                                        pctdistance=0.75, wedgeprops=dict(width=0.45))
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(9)
    
    ax2.set_title('Market Concentration', fontsize=13, fontweight='bold',
                 color=COLORS['primary'], pad=10)
    
    # Center text
    ax2.text(0, 0, '125\nHospitals', ha='center', va='center', fontsize=14,
            fontweight='bold', color=COLORS['primary'])
    
    fig.suptitle('Regional Distribution of Private Hospitals', fontsize=15,
                fontweight='bold', color=COLORS['primary'], y=1.02)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'regional_distribution.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Regional distribution chart")


# ── Chart 7: Pricing Scenarios ──────────────────────────────────────────────
def generate_pricing_scenarios():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    
    # Per encounter model
    per_encounter = [3.2, 12.1, 26.4, 43.8, 58.0]
    per_bed = [2.8, 9.5, 20.1, 33.2, 43.0]
    enterprise = [5.5, 16.8, 34.2, 55.0, 72.0]
    
    x = np.arange(len(years))
    width = 0.25
    
    bars1 = ax.bar(x - width, per_encounter, width, label='Per Encounter (SAR 8-15/enc)',
                   color=SCENARIO_COLORS[0], edgecolor='white', zorder=3)
    bars2 = ax.bar(x, per_bed, width, label='Per Bed (SAR 500-1,200/bed/mo)',
                   color=SCENARIO_COLORS[1], edgecolor='white', zorder=3)
    bars3 = ax.bar(x + width, enterprise, width, label='Enterprise License (SAR 1.5-4M/yr)',
                   color=SCENARIO_COLORS[2], edgecolor='white', zorder=3)
    
    # Value labels on Year 5
    for bars, vals in [(bars1, per_encounter), (bars2, per_bed), (bars3, enterprise)]:
        bar = bars[-1]
        val = vals[-1]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'SAR {val}M', ha='center', va='bottom', fontsize=9,
               fontweight='bold', color=COLORS['text'])
    
    ax.set_ylabel('Annual Revenue (SAR Million)', fontweight='bold')
    ax.set_title('Revenue Projection by Pricing Model', fontsize=14, fontweight='bold',
                color=COLORS['primary'], pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, max(enterprise) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, frameon=True, edgecolor=COLORS['grid'], loc='upper left')
    
    # Recommendation callout
    ax.annotate('Recommended: Hybrid\n(Per Encounter + Per Bed)',
               xy=(4, 58), fontsize=10, fontweight='bold', color=COLORS['accent'],
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E8', edgecolor=COLORS['accent']))
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'pricing_scenarios.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Pricing scenarios chart")


# ── Chart 8: Revenue Forecast ───────────────────────────────────────────────
def generate_revenue_forecast():
    forecast = DATA['revenue_forecast']
    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    hospitals = [forecast[f'year_{i}']['hospitals'] for i in range(1, 6)]
    arr = [forecast[f'year_{i}']['arr_sar_m'] for i in range(1, 6)]
    
    fig, ax1 = plt.subplots(figsize=(11, 6))
    
    # Bar: Revenue
    bars = ax1.bar(years, arr, color=COLORS['secondary'], width=0.5, edgecolor='white',
                   zorder=3, alpha=0.9)
    bars[-1].set_color(COLORS['accent'])
    
    for bar, val in zip(bars, arr):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'SAR {val}M', ha='center', va='bottom', fontsize=11,
                fontweight='bold', color=COLORS['text'])
    
    ax1.set_ylabel('Annual Recurring Revenue (SAR Million)', fontweight='bold', color=COLORS['secondary'])
    ax1.set_ylim(0, max(arr) * 1.25)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Line: Hospital count
    ax2 = ax1.twinx()
    ax2.plot(years, hospitals, 'o-', color=COLORS['accent'], linewidth=2.5, markersize=8, zorder=4)
    for i, (yr, h) in enumerate(zip(years, hospitals)):
        ax2.text(i, h + 2.5, f'{h}', ha='center', fontsize=10, fontweight='bold',
                color=COLORS['accent'])
    
    ax2.set_ylabel('Active Hospitals', fontweight='bold', color=COLORS['accent'])
    ax2.set_ylim(0, max(hospitals) * 1.3)
    ax2.spines['top'].set_visible(False)
    
    ax1.set_title('5-Year Revenue Forecast & Hospital Adoption', fontsize=14, fontweight='bold',
                 color=COLORS['primary'], pad=15)
    
    # Growth rates
    growths = ['—', '252%', '113%', '65%', '41%']
    for i, g in enumerate(growths):
        if g != '—':
            ax1.text(i, -5, f'↑{g}', ha='center', fontsize=9, color=COLORS['success'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'revenue_forecast.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Revenue forecast chart")


# ── Chart 9: Unit Economics ─────────────────────────────────────────────────
def generate_unit_economics():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    metrics = [
        ('CAC', 'SAR 180K', 180, 'Cost to acquire\none hospital', COLORS['accent']),
        ('ACV', 'SAR 1.02M', 1020, 'Average contract\nvalue per year', COLORS['secondary']),
        ('LTV', 'SAR 4.1M', 4100, 'Lifetime value\n(4-year avg.)', COLORS['success']),
        ('LTV:CAC', '22.8x', 22.8, 'Strong unit\neconomics', COLORS['primary']),
        ('Gross Margin', '82%', 82, 'Software-like\nmargins', COLORS['success']),
        ('NRR', '135%', 135, 'Net revenue\nretention', COLORS['accent2']),
    ]
    
    for ax, (name, value, num, desc, color) in zip(axes.flat, metrics):
        ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=26,
               fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.35, name, ha='center', va='center', fontsize=14,
               fontweight='bold', color=COLORS['text'], transform=ax.transAxes)
        ax.text(0.5, 0.15, desc, ha='center', va='center', fontsize=9,
               color=COLORS['muted'], transform=ax.transAxes)
        
        # Subtle background card
        rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, transform=ax.transAxes,
                              boxstyle='round,pad=0.02', facecolor=COLORS['light'],
                              edgecolor=color, linewidth=2, alpha=0.3)
        ax.add_patch(rect)
        ax.axis('off')
    
    fig.suptitle('Unit Economics Summary', fontsize=16, fontweight='bold',
                color=COLORS['primary'], y=1.02)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'unit_economics.png'), dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print("  ✓ Unit economics chart")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🏥 Saudi Healthtech Market Analysis — Generating Charts\n")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    generate_tam_sam_som()
    generate_market_growth()
    generate_denial_economics()
    generate_competitive_radar()
    generate_hospital_segmentation()
    generate_regional_distribution()
    generate_pricing_scenarios()
    generate_revenue_forecast()
    generate_unit_economics()
    
    print(f"\n✅ All charts generated successfully in {OUTPUT_DIR}/")
    print(f"   Files: {len(os.listdir(OUTPUT_DIR))} visualizations\n")
