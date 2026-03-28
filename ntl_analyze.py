"""
Nighttime Lights Analysis Tool
==============================

A reusable library for downloading and analyzing VIIRS satellite nighttime
lights data for any geographic region. Built for studying economic activity
patterns across Indian regions.

Provides functions for:
- Downloading VIIRS DNB monthly composites from Google Earth Engine
- Extracting radiance statistics within bounding boxes
- Generating time series, seasonal, and trend charts
- Comparing baseline vs recent periods

Usage as a script:
    python ntl_analyze.py --regions srinagar leh manali --download --periods 2017-2019 2023-2026

Usage as a library:
    from ntl_analyze import REGIONS, process_region, add_season_columns, plot_all_charts

Regions can be added by editing the REGIONS dictionary below.
"""

import argparse
import os
import sys
import glob
import time
import warnings
from datetime import datetime

import ee
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Region definitions ──────────────────────────────────────────────
# Add new regions here: name -> (bbox [west, south, east, north], description)
REGIONS = {
    'srinagar': {
        'bbox': [74.65, 33.9, 75.05, 34.35],
        'label': 'Srinagar (Kashmir)',
        'color': '#2196F3',
        'marker': 'o',
        'events': [
            {'date': datetime(2019, 8, 5), 'label': 'Article 370 revoked', 'color': 'red'},
        ],
    },
    'leh': {
        'bbox': [77.0, 33.8, 77.8, 34.4],
        'label': 'Leh (Ladakh)',
        'color': '#FF9800',
        'marker': 's',
        'events': [
            {'date': datetime(2019, 8, 5), 'label': 'Article 370 revoked', 'color': 'red'},
        ],
    },
    'manali': {
        'bbox': [77.05, 32.15, 77.3, 32.4],
        'label': 'Manali (Himachal)',
        'color': '#4CAF50',
        'marker': '^',
    },
    'ayodhya': {
        'bbox': [82.1, 26.72, 82.25, 26.85],
        'label': 'Ayodhya (UP)',
        'color': '#FF5722',
        'marker': '*',
        'events': [
            {'date': datetime(2020, 8, 5), 'label': 'Ram Temple Bhoomi Pujan', 'color': '#FF5722'},
            {'date': datetime(2024, 1, 22), 'label': 'Ram Temple Inauguration', 'color': 'red'},
        ],
    },
    'pune': {
        'bbox': [73.72, 18.42, 73.98, 18.62],
        'label': 'Pune (City)',
        'color': '#9C27B0',
        'marker': 'D',
    },
    'sinhgad_road': {
        'bbox': [73.78, 18.43, 73.86, 18.52],
        'label': 'Sinhgad Road (Pune)',
        'color': '#E91E63',
        'marker': 'v',
    },
    'shrivardhan': {
        'bbox': [72.93, 17.98, 73.07, 18.08],
        'label': 'Shrivardhan (Raigad)',
        'color': '#00BCD4',
        'marker': 'p',
    },
    # ── Add more regions below ──
    # To add event markers, include 'events' list:
    #   'events': [{'date': datetime(2020,1,1), 'label': 'Event', 'color': 'red'}]
    # To shade COVID gap, add 'shade_covid': True
}

# ── Seasons for India (tourism/weather angle) ──
SEASONS = {
    'Spring (Mar-May)':   [3, 4, 5],
    'Summer (Jun-Aug)':   [6, 7, 8],
    'Autumn (Sep-Nov)':   [9, 10, 11],
    'Winter (Dec-Feb)':   [12, 1, 2],
}

# Tourism seasons for hill stations / Kashmir / Ladakh
TOURISM_SEASONS = {
    'Peak Tourist (Apr-Jun)':     [4, 5, 6],
    'Monsoon (Jul-Sep)':          [7, 8, 9],
    'Autumn Tourist (Oct-Nov)':   [10, 11],
    'Winter (Dec-Mar)':           [12, 1, 2, 3],
}

# ── Paths ───────────────────────────────────────────────────────────
# Replace with your own GEE project ID if downloading data
PROJECT_ID = 'YOUR-GEE-PROJECT-ID'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RASTER_DIR = os.path.join(BASE_DIR, 'data', 'raster')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def get_monthly_dates(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = []
    current = start
    while current <= end:
        dates.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return dates


def download_viirs_month(year, month, bbox, output_dir, tag=''):
    """Download VIIRS monthly composite (radiance + cloud-free coverage)."""
    prefix = f'{tag}_' if tag else ''
    rad_filename = f'viirs_{prefix}{year}_{month:02d}.tif'
    cf_filename = f'viirs_{prefix}{year}_{month:02d}_cf.tif'
    rad_filepath = os.path.join(output_dir, rad_filename)
    cf_filepath = os.path.join(output_dir, cf_filename)

    if os.path.exists(rad_filepath) and os.path.exists(cf_filepath):
        return rad_filepath

    start = f'{year}-{month:02d}-01'
    if month == 12:
        end = f'{year + 1}-01-01'
    else:
        end = f'{year}-{month + 1:02d}-01'

    collection = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
        .filterDate(start, end)

    if collection.size().getInfo() == 0:
        print(f'  No data for {year}-{month:02d}')
        return None

    image = collection.first()
    region = ee.Geometry.Rectangle(bbox)
    dl_params = {'scale': 500, 'region': region, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326'}

    # Download radiance
    if not os.path.exists(rad_filepath):
        url = image.select('avg_rad').getDownloadURL(dl_params)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(rad_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # Download cloud-free coverage
    if not os.path.exists(cf_filepath):
        url = image.select('cf_cvg').getDownloadURL(dl_params)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(cf_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    size_kb = os.path.getsize(rad_filepath) / 1024
    print(f'  Downloaded: {rad_filename} + cf ({size_kb:.0f} KB)')
    return rad_filepath


def download_region(region_name, periods):
    """Download VIIRS data for a region across specified periods."""
    region = REGIONS[region_name]
    bbox = region['bbox']

    # Use a wider bbox for download to ensure coverage
    margin = 0.1
    dl_bbox = [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin]

    all_dates = []
    for period in periods:
        start_year, end_year = period
        all_dates.extend(get_monthly_dates(f'{start_year}-01-01', f'{end_year}-12-31'))

    print(f'\nDownloading {region["label"]}: {len(all_dates)} months')

    for year, month in all_dates:
        try:
            download_viirs_month(year, month, dl_bbox, RASTER_DIR, tag=region_name)
            time.sleep(0.5)
        except Exception as e:
            print(f'  ERROR {year}-{month:02d}: {e}')
            time.sleep(2)


def extract_radiance(tif_path, region_geom):
    """Extract radiance statistics within a region polygon.
    Also reads cloud-free coverage if available."""
    with rasterio.open(tif_path) as src:
        try:
            out_image, _ = mask(src, [mapping(region_geom)], crop=True)
            data = out_image[0]
            data = np.where(data < 0, 0, data)
            valid = data[data > 0]

            if len(valid) == 0:
                return {'mean': 0, 'sum': 0, 'median': 0, 'pixels': 0,
                        'max': 0, 'cf_mean': 0, 'cf_quality': 'no_data'}

            result = {
                'mean': float(np.mean(valid)),
                'sum': float(np.sum(valid)),
                'median': float(np.median(valid)),
                'pixels': int(len(valid)),
                'max': float(np.max(valid)),
            }

            # Try to read cloud-free coverage file
            cf_path = tif_path.replace('.tif', '_cf.tif')
            if os.path.exists(cf_path):
                with rasterio.open(cf_path) as cf_src:
                    cf_image, _ = mask(cf_src, [mapping(region_geom)], crop=True)
                    cf_data = cf_image[0]
                    cf_valid = cf_data[cf_data >= 0]
                    cf_mean = float(np.mean(cf_valid)) if len(cf_valid) > 0 else 0
                    result['cf_mean'] = cf_mean
                    # Quality flag: monsoon months often have cf < 5
                    if cf_mean < 3:
                        result['cf_quality'] = 'poor'
                    elif cf_mean < 8:
                        result['cf_quality'] = 'fair'
                    else:
                        result['cf_quality'] = 'good'
            else:
                result['cf_mean'] = -1  # not available
                result['cf_quality'] = 'unknown'

            return result
        except Exception as e:
            print(f'Warning: failed to extract radiance from {tif_path}: {e}')
            return None


def process_region(region_name):
    """Process all downloaded files for a region."""
    region = REGIONS[region_name]
    region_geom = box(*region['bbox'])

    # Collect region-specific files
    region_files = sorted([f for f in glob.glob(os.path.join(RASTER_DIR, f'viirs_{region_name}_*.tif'))
                           if not f.endswith('_cf.tif')])

    # Also collect combined/generic files (backward compat for srinagar/leh)
    combined_files = sorted([f for f in glob.glob(os.path.join(RASTER_DIR, 'viirs_[0-9]*.tif'))
                             if not f.endswith('_cf.tif')])

    # Build a dict of (year, month) -> filepath, region-specific takes priority
    file_map = {}
    for tif_path in combined_files:
        parts = os.path.basename(tif_path).replace('.tif', '').split('_')
        key = (int(parts[-2]), int(parts[-1]))
        file_map[key] = tif_path

    for tif_path in region_files:
        parts = os.path.basename(tif_path).replace('.tif', '').split('_')
        key = (int(parts[-2]), int(parts[-1]))
        file_map[key] = tif_path  # overrides combined if both exist

    if not file_map:
        print(f'No data files found for {region_name}')
        return pd.DataFrame()

    tif_files = [file_map[k] for k in sorted(file_map.keys())]

    records = []
    for tif_path in tif_files:
        fname = os.path.basename(tif_path).replace('.tif', '')
        parts = fname.split('_')
        year, month = int(parts[-2]), int(parts[-1])

        stats = extract_radiance(tif_path, region_geom)
        if stats:
            records.append({
                'date': datetime(year, month, 1),
                'year': year,
                'month': month,
                'region': region['label'],
                'region_key': region_name,
                **stats,
            })

    return pd.DataFrame(records)


def add_season_columns(df):
    """Add season and tourism season columns."""
    def get_season(month):
        for name, months in SEASONS.items():
            if month in months:
                return name
        return 'Unknown'

    def get_tourism_season(month):
        for name, months in TOURISM_SEASONS.items():
            if month in months:
                return name
        return 'Unknown'

    df['season'] = df['month'].apply(get_season)
    df['tourism_season'] = df['month'].apply(get_tourism_season)
    def classify_period(d):
        if d.year <= 2016:
            return 'Early (2014-2016)'
        elif d < datetime(2019, 8, 1):
            return 'Pre-Art.370 (2017-Jul 2019)'
        elif d < datetime(2020, 1, 1):
            return 'Post-Art.370 (Aug-Dec 2019)'
        elif d < datetime(2023, 1, 1):
            return 'COVID era (2020-2022)'
        else:
            return 'Post-COVID (2023-2026)'

    df['period'] = df['date'].apply(classify_period)
    return df


# ── Helper for event markers ────────────────────────────────────────

def add_event_markers(ax, rkey, date_axis=True):
    """Add region-specific event markers and COVID shading to an axis."""
    rinfo = REGIONS[rkey]
    events = rinfo.get('events', [])
    shade_covid = rinfo.get('shade_covid', False)

    for evt in events:
        if date_axis:
            ax.axvline(x=evt['date'], color=evt.get('color', 'red'),
                       linestyle='--', linewidth=2, label=evt['label'])
        else:
            # For yearly x-axis (float years)
            yr_frac = evt['date'].year + evt['date'].month / 12
            ax.axvline(x=yr_frac, color=evt.get('color', 'red'),
                       linestyle='--', linewidth=2, label=evt['label'])

    if shade_covid:
        if date_axis:
            ax.axvspan(datetime(2020, 1, 1), datetime(2022, 12, 31),
                       alpha=0.12, color='gray', label='COVID gap')
        else:
            ax.axvspan(2020, 2022, alpha=0.12, color='gray', label='COVID gap')


def get_baseline_and_labels(rdf):
    """Determine baseline period and labels based on available data."""
    early = rdf[rdf['period'] == 'Early (2014-2016)']
    pre = rdf[rdf['period'] == 'Pre-Art.370 (2017-Jul 2019)']
    post = rdf[rdf['period'] == 'Post-COVID (2023-2026)']

    if not early.empty:
        baseline = early
        bl_label = f'Early ({int(early["year"].min())}-{int(early["year"].max())})'
    elif not pre.empty:
        baseline = pre
        bl_label = 'Pre (2017-2019)'
    else:
        baseline = rdf[rdf['year'] <= rdf['year'].median()]
        bl_label = 'Earlier'

    post_label = f'Recent ({int(post["year"].min())}-{int(post["year"].max())})' if not post.empty else 'Recent'
    return baseline, post, bl_label, post_label


# ── Plotting functions (per-region) ─────────────────────────────────

def plot_all_charts(df, regions):
    """Generate all charts -- one set of images per region."""
    for rkey in regions:
        rinfo = REGIONS[rkey]
        rdf = df[df['region_key'] == rkey].sort_values('date').copy()
        if rdf.empty:
            continue

        plot_timeseries_single(rdf, rkey, rinfo)
        plot_seasonal_single(rdf, rkey, rinfo)
        plot_tourism_season_single(rdf, rkey, rinfo)
        plot_yearly_trend_single(rdf, rkey, rinfo)
        plot_winter_vs_summer_single(rdf, rkey, rinfo)


def plot_timeseries_single(rdf, rkey, rinfo):
    """Time series for a single region."""
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(rdf['date'], rdf['sum'], color=rinfo['color'], linewidth=1, alpha=0.4)
    rdf_ma = rdf.copy()
    rdf_ma['sum_ma'] = rdf_ma['sum'].rolling(window=3, min_periods=1).mean()
    ax.plot(rdf_ma['date'], rdf_ma['sum_ma'], color=rinfo['color'], linewidth=2.5,
            label=f'{rinfo["label"]} (3-month avg)')

    # Seasonal shading
    for year in rdf['year'].unique():
        ax.axvspan(datetime(year, 4, 1), datetime(year, 6, 30),
                   alpha=0.08, color='gold')
        if datetime(year, 12, 1) <= rdf['date'].max():
            end_yr = min(year + 1, rdf['year'].max() + 1)
            ax.axvspan(datetime(year, 12, 1), datetime(end_yr, 2, 28),
                       alpha=0.08, color='lightblue')

    add_event_markers(ax, rkey)

    ax.set_ylabel('Total Radiance', fontsize=12)
    ax.set_title(f'{rinfo["label"]} -- Nighttime Lights Time Series', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.text(0.98, 0.02, 'Yellow=Summer (Apr-Jun)  Blue=Winter (Dec-Feb)',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            style='italic', color='gray')

    plt.tight_layout()
    fname = f'timeseries_{rkey}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_seasonal_single(rdf, rkey, rinfo):
    """Monthly pattern comparison for a single region."""
    baseline, post, bl_label, post_label = get_baseline_and_labels(rdf)

    if baseline.empty or post.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    bl_monthly = baseline.groupby('month')['sum'].mean()
    post_monthly = post.groupby('month')['sum'].mean()

    x = np.arange(12)
    width = 0.35
    bl_vals = [bl_monthly.get(m, 0) for m in range(1, 13)]
    post_vals = [post_monthly.get(m, 0) for m in range(1, 13)]

    ax.bar(x - width/2, bl_vals, width, label=bl_label,
           color='#90CAF9', edgecolor='#1565C0')
    ax.bar(x + width/2, post_vals, width, label=post_label,
           color='#FFB74D', edgecolor='#E65100')

    for i, (bv, pv) in enumerate(zip(bl_vals, post_vals)):
        if bv > 0:
            pct = ((pv - bv) / bv) * 100
            ax.text(x[i] + width/2, pv + max(post_vals) * 0.02,
                    f'{pct:+.0f}%', ha='center', va='bottom', fontsize=7,
                    fontweight='bold', color='#E65100')

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    for i, label in enumerate(ax.get_xticklabels()):
        m = i + 1
        if m in [6, 7, 8, 9]:       # Monsoon
            label.set_color('#2E7D32')
            label.set_fontweight('bold')
        elif m in [12, 1, 2]:        # Winter
            label.set_color('#1565C0')

    ax.set_xlabel('Month (green=monsoon, blue=winter)', fontsize=10)
    ax.set_ylabel('Avg Total Radiance', fontsize=12)
    ax.set_title(f'{rinfo["label"]} -- Monthly Pattern', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f'seasonal_pattern_{rkey}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_tourism_season_single(rdf, rkey, rinfo):
    """Tourism season comparison for a single region."""
    baseline, post, bl_label, post_label = get_baseline_and_labels(rdf)
    if baseline.empty or post.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    season_order = list(TOURISM_SEASONS.keys())
    season_colors = ['#FF9800', '#66BB6A', '#EF5350', '#42A5F5']

    x = np.arange(len(season_order))
    width = 0.35

    bl_vals = [baseline[baseline['tourism_season'] == s]['sum'].mean() for s in season_order]
    post_vals = [post[post['tourism_season'] == s]['sum'].mean() for s in season_order]

    ax.bar(x - width/2, bl_vals, width, label=bl_label,
           color=[c + '80' for c in season_colors], edgecolor=season_colors)
    ax.bar(x + width/2, post_vals, width, label=post_label,
           color=season_colors, edgecolor='black', linewidth=0.5)

    for i, (bv, pv) in enumerate(zip(bl_vals, post_vals)):
        if bv > 0:
            pct = ((pv - bv) / bv) * 100
            ax.text(x[i] + width/2, pv + max(post_vals) * 0.02,
                    f'{pct:+.0f}%', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    short_labels = ['Peak Tourist\n(Apr-Jun)', 'Monsoon\n(Jul-Sep)',
                    'Autumn\n(Oct-Nov)', 'Winter\n(Dec-Mar)']
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel('Avg Total Radiance', fontsize=12)
    ax.set_title(f'{rinfo["label"]} -- By Season', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f'tourism_season_{rkey}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_yearly_trend_single(rdf, rkey, rinfo):
    """Yearly trend for a single region."""
    fig, ax = plt.subplots(figsize=(12, 6))

    yearly = rdf.groupby('year')['sum'].mean()
    ax.plot(yearly.index, yearly.values, color=rinfo['color'],
            marker=rinfo['marker'], linewidth=2.5, markersize=8,
            label=rinfo['label'])

    add_event_markers(ax, rkey, date_axis=False)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Total Radiance', fontsize=12)
    ax.set_title(f'{rinfo["label"]} -- Yearly Average Nighttime Lights', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(rdf['year'].unique()))

    plt.tight_layout()
    fname = f'yearly_trend_{rkey}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_winter_vs_summer_single(rdf, rkey, rinfo):
    """Winter vs summer comparison for a single region."""
    fig, ax = plt.subplots(figsize=(10, 6))

    summer = rdf[rdf['month'].isin([4, 5, 6])].groupby('year')['sum'].mean()
    winter = rdf[rdf['month'].isin([12, 1, 2])].groupby('year')['sum'].mean()

    years = sorted(set(summer.index) | set(winter.index))
    x = np.arange(len(years))
    width = 0.35

    s_vals = [summer.get(y, 0) for y in years]
    w_vals = [winter.get(y, 0) for y in years]

    ax.bar(x - width/2, s_vals, width, label='Summer (Apr-Jun)',
           color='#FF9800', edgecolor='#E65100')
    ax.bar(x + width/2, w_vals, width, label='Winter (Dec-Feb)',
           color='#42A5F5', edgecolor='#1565C0')

    for i, (s, w) in enumerate(zip(s_vals, w_vals)):
        if w > 0 and s > 0:
            ratio = s / w
            ax.text(x[i], max(s, w) + max(max(s_vals), max(w_vals)) * 0.03,
                    f'{ratio:.1f}x', ha='center', fontsize=8, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.set_ylabel('Avg Total Radiance', fontsize=12)
    ax.set_title(f'{rinfo["label"]} -- Summer vs Winter', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f'winter_vs_summer_{rkey}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def print_summary(df, regions):
    """Print detailed summary statistics."""
    print('\n' + '=' * 75)
    print('SUMMARY STATISTICS -- NIGHTTIME LIGHTS ANALYSIS')
    print('=' * 75)

    for rkey in regions:
        rinfo = REGIONS[rkey]
        rdf = df[df['region_key'] == rkey]

        # Use earliest and latest available periods for comparison
        early = rdf[rdf['year'] <= 2016]
        pre = rdf[rdf['period'] == 'Pre-Art.370 (2017-Jul 2019)']
        post = rdf[rdf['period'] == 'Post-COVID (2023-2026)']

        # Pick the best baseline: early (2014-2016) if available, else pre (2017-2019)
        if not early.empty:
            baseline = early
            baseline_label = f'Early ({early["year"].min()}-{early["year"].max()})'
        elif not pre.empty:
            baseline = pre
            baseline_label = 'Pre-Art.370 (2017-Jul 2019)'
        else:
            print(f'\n--- {rinfo["label"]}: insufficient data ---')
            continue

        if post.empty:
            print(f'\n--- {rinfo["label"]}: no post-COVID data ---')
            continue

        base_mean = baseline['sum'].mean()
        post_mean = post['sum'].mean()
        change = ((post_mean - base_mean) / base_mean) * 100

        print(f'\n{"+" * 60}')
        print(f'  {rinfo["label"]}')
        print(f'{"+" * 60}')
        print(f'  Overall change ({baseline_label} -> Post-COVID): {change:+.1f}%')
        print(f'    Baseline: {base_mean:>10,.0f}    Post: {post_mean:>10,.0f}')

        # Monsoon quality check
        if 'cf_mean' in rdf.columns:
            monsoon = rdf[rdf['month'].isin([6, 7, 8, 9])]
            non_monsoon = rdf[~rdf['month'].isin([6, 7, 8, 9])]
            if not monsoon.empty and 'cf_mean' in monsoon.columns:
                m_cf = monsoon['cf_mean'].mean()
                nm_cf = non_monsoon['cf_mean'].mean()
                if m_cf >= 0:
                    print(f'\n  Cloud-Free Observations (avg):')
                    print(f'    Monsoon (Jun-Sep):     {m_cf:.1f}')
                    print(f'    Non-monsoon:           {nm_cf:.1f}')
                    if m_cf < 5:
                        print(f'    WARNING: Monsoon data quality is LOW -- interpret Jun-Sep with caution')

        # By season
        print(f'\n  By Season:')
        print(f'    {"Season":<25} {"Baseline":>10} {"Post":>10} {"Change":>10}')
        print(f'    {"-" * 55}')
        for sname in TOURISM_SEASONS:
            s_base = baseline[baseline['tourism_season'] == sname]['sum'].mean()
            s_post = post[post['tourism_season'] == sname]['sum'].mean()
            if s_base > 0:
                s_change = ((s_post - s_base) / s_base) * 100
                print(f'    {sname:<25} {s_base:>10,.0f} {s_post:>10,.0f} {s_change:>+9.1f}%')

        # Year-over-year
        print(f'\n  Year-over-Year:')
        for year in sorted(rdf['year'].unique()):
            yr = rdf[rdf['year'] == year]
            cf_str = ''
            if 'cf_mean' in yr.columns:
                cf_avg = yr['cf_mean'].mean()
                if cf_avg >= 0:
                    cf_str = f'  (avg cloud-free obs: {cf_avg:.1f})'
            print(f'    {year}: {yr["sum"].mean():>10,.0f}{cf_str}')

    # Seasonal ratio analysis
    print(f'\n{"=" * 75}')
    print('SEASONAL RATIO: Summer(Apr-Jun) / Winter(Dec-Feb)')
    print(f'{"=" * 75}')
    print(f'  {"Region":<25} {"Early/Pre":>12} {"Post":>12} {"Interpretation"}')
    print(f'  {"-" * 70}')

    for rkey in regions:
        rinfo = REGIONS[rkey]
        rdf = df[df['region_key'] == rkey]

        early = rdf[rdf['year'] <= 2016]
        pre = rdf[rdf['period'] == 'Pre-Art.370 (2017-Jul 2019)']
        baseline = early if not early.empty else pre
        post = rdf[rdf['period'] == 'Post-COVID (2023-2026)']

        if baseline.empty or post.empty:
            continue

        base_summer = baseline[baseline['month'].isin([4, 5, 6])]['sum'].mean()
        base_winter = baseline[baseline['month'].isin([12, 1, 2])]['sum'].mean()
        post_summer = post[post['month'].isin([4, 5, 6])]['sum'].mean()
        post_winter = post[post['month'].isin([12, 1, 2])]['sum'].mean()

        base_ratio = base_summer / base_winter if base_winter > 0 else 0
        post_ratio = post_summer / post_winter if post_winter > 0 else 0

        if post_ratio < base_ratio:
            interp = 'Winter catching up (year-round activity)'
        else:
            interp = 'Summer growing faster'

        print(f'  {rinfo["label"]:<25} {base_ratio:>10.2f}x  {post_ratio:>10.2f}x  {interp}')


def main():
    parser = argparse.ArgumentParser(description='VIIRS Nighttime Lights Analysis')
    parser.add_argument('--regions', nargs='+', default=['srinagar', 'leh'],
                        choices=list(REGIONS.keys()),
                        help='Regions to analyze')
    parser.add_argument('--download', action='store_true',
                        help='Download data (skip if already downloaded)')
    parser.add_argument('--periods', nargs='+', default=['2017-2019', '2023-2026'],
                        help='Year ranges (e.g., 2017-2019 2023-2026)')

    args = parser.parse_args()

    os.makedirs(RASTER_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse periods
    periods = []
    for p in args.periods:
        start, end = p.split('-')
        periods.append((int(start), int(end)))

    # Download if requested
    if args.download:
        ee.Initialize(project=PROJECT_ID)
        for rkey in args.regions:
            download_region(rkey, periods)

    # Process all regions
    print('\n=== Processing regions ===')
    all_dfs = []
    for rkey in args.regions:
        print(f'Processing {REGIONS[rkey]["label"]}...')
        rdf = process_region(rkey)
        if not rdf.empty:
            all_dfs.append(rdf)
            print(f'  {len(rdf)} records')

    if not all_dfs:
        print('No data to analyze!')
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df = add_season_columns(df)
    df.to_csv(os.path.join(OUTPUT_DIR, 'radiance_all_regions.csv'), index=False)

    # Generate per-region plots
    print('\n=== Generating plots (per region) ===')
    plot_all_charts(df, args.regions)

    # Print summary
    print_summary(df, args.regions)

    print(f'\nAll outputs saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
