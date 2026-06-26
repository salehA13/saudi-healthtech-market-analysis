<div align="center">

# 🏥 Saudi Healthtech Market Sizing

![CI](https://github.com/salehA13/saudi-healthtech-market-analysis/actions/workflows/ci.yml/badge.svg)

### Exploring the numbers behind Saudi Arabia's healthcare transformation

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

*A data visualization project exploring Saudi Arabia's healthcare market — hospital segmentation, regional distribution, market sizing, and financial modeling. Built as a playground for market analysis techniques.*

<br>

<img src="output/tam_sam_som.png" width="650" alt="TAM SAM SOM Analysis"/>

</div>

---

## What this is

A weekend project practicing market sizing and financial modeling on publicly available Saudi healthcare data. Generates publication-quality charts using Python + Matplotlib. Not a business document — just charts and code.

---

## 📊 Charts

<table>
<tr>
<td align="center" width="50%">
<img src="output/tam_sam_som.png" width="100%" alt="TAM SAM SOM"/><br>
<b>1. Market Sizing</b><br>
<sub>Concentric TAM/SAM/SOM visualization</sub>
</td>
<td align="center" width="50%">
<img src="output/market_growth.png" width="100%" alt="Market Growth"/><br>
<b>2. Healthcare Market Projection</b><br>
<sub>Saudi healthcare expenditure at 6.7% CAGR</sub>
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="output/hospital_segmentation.png" width="100%" alt="Hospital Segmentation"/><br>
<b>3. Hospital Tier Segmentation</b><br>
<sub>125 private hospitals by tier, scale, and revenue</sub>
</td>
<td align="center" width="50%">
<img src="output/regional_distribution.png" width="100%" alt="Regional Distribution"/><br>
<b>4. Regional Distribution</b><br>
<sub>Geographic concentration across KSA regions</sub>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="output/pricing_scenarios.png" width="100%" alt="Pricing Scenarios"/><br>
<b>5. Pricing Scenarios</b><br>
<sub>Revenue projection across pricing models</sub>
</td>
<td align="center" width="33%">
<img src="output/revenue_forecast.png" width="100%" alt="Revenue Forecast"/><br>
<b>6. Revenue Forecast</b><br>
<sub>5-year ARR trajectory with adoption curve</sub>
</td>
<td align="center" width="34%">
<img src="output/unit_economics.png" width="100%" alt="Unit Economics"/><br>
<b>7. Unit Economics</b><br>
<sub>Standard SaaS metrics dashboard</sub>
</td>
</tr>
</table>

---

## 🔬 Methodology

| Approach | Method |
|----------|--------|
| **Market Sizing** | Top-down from macro indicators validated with bottom-up hospital economics |
| **Segmentation** | K-means clustering on bed count, revenue, and digital maturity |
| **Financial Modeling** | DCF with Monte Carlo simulation across 3 pricing scenarios |

### Data Sources

| Source | Type |
|--------|------|
| Saudi Ministry of Health Statistical Yearbook | Public government data |
| Council of Health Insurance (CHI) Annual Report | Regulatory filings |
| Tadawul Capital Market Filings | Public financial disclosures |
| Frost & Sullivan — MENA Health IT Report | Industry research |

---

## 🚀 Quick Start

```bash
git clone https://github.com/salehA13/saudi-healthtech-market-analysis.git
cd saudi-healthtech-market-analysis
pip install -r requirements.txt
python src/generate_all_charts.py
```

---

<div align="center">

*By [Saleh Alkhudairy](https://github.com/salehA13) — a data viz playground*

</div>
