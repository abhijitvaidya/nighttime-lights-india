"""
Correlate nighttime lights data with official J&K economic indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def main():
    # ── Load nighttime lights data ──
    ntl = pd.read_csv(os.path.join(OUTPUT_DIR, 'radiance_all_regions.csv'))
    ntl['date'] = pd.to_datetime(ntl['date'])

    # Compute fiscal year averages for Srinagar (Apr-Mar)
    sri = ntl[ntl['region_key'] == 'srinagar'].copy()
    sri['fiscal_year'] = sri.apply(
        lambda r: f"{r['year']}-{str(r['year']+1)[2:]}" if r['month'] >= 4
        else f"{r['year']-1}-{str(r['year'])[2:]}", axis=1)
    sri_fy = sri.groupby('fiscal_year')['sum'].mean().reset_index()
    sri_fy.columns = ['fiscal_year', 'ntl_radiance']

    # Same for Leh
    leh = ntl[ntl['region_key'] == 'leh'].copy()
    leh['fiscal_year'] = leh.apply(
        lambda r: f"{r['year']}-{str(r['year']+1)[2:]}" if r['month'] >= 4
        else f"{r['year']-1}-{str(r['year'])[2:]}", axis=1)
    leh_fy = leh.groupby('fiscal_year')['sum'].mean().reset_index()
    leh_fy.columns = ['fiscal_year', 'ntl_radiance']

    # Combined Srinagar + Leh (proxy for old undivided J&K)
    combined = ntl[ntl['region_key'].isin(['srinagar', 'leh'])].copy()
    combined['fiscal_year'] = combined.apply(
        lambda r: f"{r['year']}-{str(r['year']+1)[2:]}" if r['month'] >= 4
        else f"{r['year']-1}-{str(r['year'])[2:]}", axis=1)
    comb_fy = combined.groupby('fiscal_year')['sum'].mean().reset_index()
    comb_fy.columns = ['fiscal_year', 'ntl_radiance']

    # ── Load official economic data ──
    econ = pd.read_csv(os.path.join(BASE_DIR, 'data', 'economic', 'jk_ladakh_economic_data.csv'))

    # Extract key series
    gsdp_current = econ[econ['indicator'] == 'GSDP_current_prices'][['fiscal_year', 'value']].copy()
    gsdp_current.columns = ['fiscal_year', 'gsdp_current']

    gsdp_constant = econ[econ['indicator'] == 'GSDP_constant_2011_12'][['fiscal_year', 'value']].copy()
    gsdp_constant.columns = ['fiscal_year', 'gsdp_constant']

    per_capita = econ[econ['indicator'] == 'per_capita_GSDP_current'][['fiscal_year', 'value']].copy()
    per_capita.columns = ['fiscal_year', 'per_capita']

    power_cost = econ[econ['indicator'] == 'power_purchase_cost'][['fiscal_year', 'value']].copy()
    power_cost.columns = ['fiscal_year', 'power_cost']

    power_units = econ[econ['indicator'] == 'power_purchase_units'][['fiscal_year', 'value']].copy()
    power_units.columns = ['fiscal_year', 'power_units']

    elec = econ[econ['indicator'] == 'electricity_consumption'][['fiscal_year', 'value']].copy()
    elec.columns = ['fiscal_year', 'elec_gwh']

    tourism = econ[(econ['indicator'] == 'tourist_arrivals_total') &
                   (econ['region'].str.startswith('J&K'))][['fiscal_year', 'value']].copy()
    tourism.columns = ['fiscal_year', 'tourists']

    # ── Merge everything ──
    # Use fiscal year for GSDP/power, calendar year for tourism
    merged = comb_fy.merge(gsdp_current, on='fiscal_year', how='outer') \
                    .merge(gsdp_constant, on='fiscal_year', how='outer') \
                    .merge(per_capita, on='fiscal_year', how='outer') \
                    .merge(power_cost, on='fiscal_year', how='outer')
    merged = merged.sort_values('fiscal_year')

    # Filter to years with both NTL and economic data
    merged_valid = merged.dropna(subset=['ntl_radiance', 'gsdp_current'])

    print("=== Merged Data (NTL + Official) ===")
    print(merged_valid[['fiscal_year', 'ntl_radiance', 'gsdp_current', 'gsdp_constant',
                         'per_capita', 'power_cost']].to_string(index=False))

    # ── Compute correlations ──
    print("\n=== Correlation Coefficients ===")
    for col, label in [('gsdp_current', 'GSDP (current prices)'),
                        ('gsdp_constant', 'GSDP (constant 2011-12)'),
                        ('per_capita', 'Per capita GSDP'),
                        ('power_cost', 'Power purchase cost')]:
        valid = merged_valid.dropna(subset=[col])
        if len(valid) >= 3:
            corr = valid['ntl_radiance'].corr(valid[col])
            print(f"  NTL vs {label}: r = {corr:.3f} (n={len(valid)})")

    # ── Plot 1: NTL vs GSDP dual-axis ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: NTL vs GSDP Current
    ax1 = axes[0, 0]
    valid = merged_valid.dropna(subset=['gsdp_current'])
    x = range(len(valid))
    ax1.bar(x, valid['gsdp_current'] / 1000, color='#90CAF9', edgecolor='#1565C0',
            alpha=0.7, label='GSDP (current, Rs \'000 Cr)')
    ax1b = ax1.twinx()
    ax1b.plot(x, valid['ntl_radiance'] / 1000, color='#E53935', marker='o',
              linewidth=2.5, label='NTL Radiance (\'000)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid['fiscal_year'], rotation=45, fontsize=8)
    ax1.set_ylabel('GSDP (Rs \'000 Crore)', color='#1565C0', fontsize=11)
    ax1b.set_ylabel('NTL Radiance (\'000)', color='#E53935', fontsize=11)
    corr = valid['ntl_radiance'].corr(valid['gsdp_current'])
    ax1.set_title(f'NTL vs GSDP (Current Prices)\nr = {corr:.3f}', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1b.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Panel 2: NTL vs GSDP Constant
    ax2 = axes[0, 1]
    valid2 = merged_valid.dropna(subset=['gsdp_constant'])
    x2 = range(len(valid2))
    ax2.bar(x2, valid2['gsdp_constant'] / 1000, color='#A5D6A7', edgecolor='#2E7D32',
            alpha=0.7, label='GSDP (constant 2011-12, Rs \'000 Cr)')
    ax2b = ax2.twinx()
    ax2b.plot(x2, valid2['ntl_radiance'] / 1000, color='#E53935', marker='o',
              linewidth=2.5, label='NTL Radiance (\'000)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(valid2['fiscal_year'], rotation=45, fontsize=8)
    ax2.set_ylabel('GSDP (Rs \'000 Crore, constant)', color='#2E7D32', fontsize=11)
    ax2b.set_ylabel('NTL Radiance (\'000)', color='#E53935', fontsize=11)
    corr2 = valid2['ntl_radiance'].corr(valid2['gsdp_constant'])
    ax2.set_title(f'NTL vs GSDP (Constant Prices)\nr = {corr2:.3f}', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2b.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Panel 3: NTL vs Power Purchase Cost
    ax3 = axes[1, 0]
    valid3 = merged_valid.dropna(subset=['power_cost'])
    x3 = range(len(valid3))
    ax3.bar(x3, valid3['power_cost'], color='#FFE082', edgecolor='#F57F17',
            alpha=0.7, label='Power Purchase Cost (Rs Cr)')
    ax3b = ax3.twinx()
    ax3b.plot(x3, valid3['ntl_radiance'] / 1000, color='#E53935', marker='o',
              linewidth=2.5, label='NTL Radiance (\'000)')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(valid3['fiscal_year'], rotation=45, fontsize=8)
    ax3.set_ylabel('Power Purchase Cost (Rs Crore)', color='#F57F17', fontsize=11)
    ax3b.set_ylabel('NTL Radiance (\'000)', color='#E53935', fontsize=11)
    corr3 = valid3['ntl_radiance'].corr(valid3['power_cost'])
    ax3.set_title(f'NTL vs Power Purchase Cost\nr = {corr3:.3f}', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3b.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.2)

    # Panel 4: Scatter plot - NTL vs GSDP with regression line
    ax4 = axes[1, 1]
    valid4 = merged_valid.dropna(subset=['gsdp_current'])
    ax4.scatter(valid4['gsdp_current'] / 1000, valid4['ntl_radiance'] / 1000,
                color='#1565C0', s=100, zorder=5, edgecolors='black')

    # Add labels
    for _, row in valid4.iterrows():
        ax4.annotate(row['fiscal_year'],
                     (row['gsdp_current']/1000, row['ntl_radiance']/1000),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Regression line
    z = np.polyfit(valid4['gsdp_current']/1000, valid4['ntl_radiance']/1000, 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid4['gsdp_current'].min()/1000, valid4['gsdp_current'].max()/1000, 100)
    ax4.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit (r={corr:.3f})')

    ax4.set_xlabel('GSDP Current Prices (Rs \'000 Crore)', fontsize=11)
    ax4.set_ylabel('NTL Radiance (\'000)', fontsize=11)
    ax4.set_title('NTL vs GSDP \u2014 Scatter Plot', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Nighttime Lights vs Official Economic Data \u2014 J&K (Srinagar + Leh)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ntl_vs_official_data.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved: ntl_vs_official_data.png')

    # ── Plot 2: Normalized growth comparison ──
    fig, ax = plt.subplots(figsize=(13, 7))

    # Use the merged dataframe which has aligned fiscal years
    norm_base = merged_valid.iloc[0]
    fiscal_years = merged_valid['fiscal_year'].values
    x = range(len(fiscal_years))

    for col, label, color, marker in [
        ('ntl_radiance', 'Nighttime Lights (Srinagar+Leh)', '#E53935', 'o'),
        ('gsdp_current', 'GSDP (current prices)', '#1565C0', 's'),
        ('gsdp_constant', 'GSDP (constant 2011-12)', '#2E7D32', '^'),
        ('per_capita', 'Per capita GSDP', '#FF9800', 'D'),
        ('power_cost', 'Power purchase cost', '#9C27B0', 'v'),
    ]:
        if col not in merged_valid.columns:
            continue
        vals = merged_valid[col].values
        base_val = vals[0]
        if pd.isna(base_val) or base_val == 0:
            continue
        normalized = [(v / base_val * 100) if pd.notna(v) else np.nan for v in vals]
        ax.plot(list(x), normalized, color=color, marker=marker, linewidth=2, markersize=7, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(fiscal_years, rotation=45, fontsize=8)
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Dynamically compute the Article 370 line position based on '2019-20' in the data
    fiscal_year_list = list(fiscal_years)
    if '2019-20' in fiscal_year_list:
        art370_pos = fiscal_year_list.index('2019-20')
    else:
        art370_pos = 5  # fallback
    ax.axvline(x=art370_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Art. 370 revoked / UT formation')
    ax.set_ylabel('Index (2014-15 = 100)', fontsize=12)
    ax.set_title('Growth Comparison: Nighttime Lights vs Official Economic Indicators\nJ&K (Srinagar + Leh combined)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ntl_vs_official_normalized.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: ntl_vs_official_normalized.png')

    # ── Plot 3: Srinagar vs Leh separate comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, region_fy, title in [(axes[0], sri_fy, 'Srinagar'), (axes[1], leh_fy, 'Leh')]:
        region_merged = region_fy.merge(gsdp_current, on='fiscal_year', how='inner')
        if len(region_merged) < 3:
            continue

        base_ntl = region_merged.iloc[0]['ntl_radiance']
        base_gsdp = region_merged.iloc[0]['gsdp_current']

        ntl_norm = (region_merged['ntl_radiance'] / base_ntl) * 100
        gsdp_norm = (region_merged['gsdp_current'] / base_gsdp) * 100

        x = range(len(region_merged))
        ax.plot(x, ntl_norm, color='#E53935', marker='o', linewidth=2.5, label='NTL Radiance')
        ax.plot(x, gsdp_norm, color='#1565C0', marker='s', linewidth=2.5, label='J&K GSDP')
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(region_merged['fiscal_year'], rotation=45, fontsize=8)
        ax.set_ylabel('Index (first year = 100)', fontsize=11)
        ax.set_title(f'{title} NTL vs J&K GSDP (Normalized)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ntl_srinagar_leh_vs_gsdp.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: ntl_srinagar_leh_vs_gsdp.png')

    # ── Print summary ──
    print('\n' + '=' * 70)
    print('VALIDATION SUMMARY')
    print('=' * 70)
    print(f'\nData points with both NTL and GSDP: {len(merged_valid)}')
    print(f'\nCorrelation (NTL vs GSDP current):   r = {corr:.3f}')
    print(f'Correlation (NTL vs GSDP constant):  r = {corr2:.3f}')
    print(f'Correlation (NTL vs Power cost):     r = {corr3:.3f}')

    print('\n--- Growth Rates (2014-15 to latest) ---')
    ntl_growth = (merged_valid.iloc[-1]['ntl_radiance'] / merged_valid.iloc[0]['ntl_radiance'] - 1) * 100
    gsdp_growth = (merged_valid.iloc[-1]['gsdp_current'] / merged_valid.iloc[0]['gsdp_current'] - 1) * 100
    gsdp_real = (merged_valid.iloc[-1]['gsdp_constant'] / merged_valid.iloc[0]['gsdp_constant'] - 1) * 100
    print(f'  NTL Radiance growth:    {ntl_growth:+.1f}%')
    print(f'  GSDP (current) growth:  {gsdp_growth:+.1f}%')
    print(f'  GSDP (constant) growth: {gsdp_real:+.1f}%')

    print('\nNote: Pre-2019 GSDP includes Ladakh; post-2019 excludes it.')
    print('The NTL data covers Srinagar+Leh combined for comparability,')
    print('but the structural break in GSDP makes exact comparison imperfect.')


if __name__ == '__main__':
    main()
