# Nighttime Lights and Economic Activity: Evidence from India

**Using VIIRS satellite data to measure economic transformation across Indian regions**

Inspired by ["Shedding Light on the Russia-Ukraine War"](https://www.xkdr.org/paper/shedding-light-on-the-russia-ukraine-war) (Hande, Patnaik, Shah, Thomas — xKDR Forum Working Paper 40, 2025) and the accompanying [podcast discussion](https://www.youtube.com/watch?v=SYR0CvBzUkI), this project applies their nighttime lights methodology to Indian regions undergoing rapid policy-driven change.

## Key Findings

- **Srinagar (Kashmir):** +91% increase in nighttime radiance from 2014-2016 baseline to 2023-2025, with growth accelerating after Article 370 revocation (August 2019) and post-COVID recovery
- **Leh (Ladakh):** +225% surge in nighttime lights driven by massive infrastructure and military buildup after becoming a Union Territory, followed by a recent decline as construction activity normalizes
- **Ayodhya:** A textbook hockey-stick pattern — flat radiance for years, then a sharp upward inflection coinciding with the Ram Temple construction (Bhoomi Pujan August 2020) and inauguration (January 2024)
- **Manali:** Strong seasonal tourism signal with summer peaks 2-3x winter levels, serving as a useful control region

## Validation

Nighttime lights correlate strongly with official J&K economic indicators:
- GSDP at current prices: r = 0.97
- GSDP at constant prices: r = 0.93
- Power purchase costs: r = 0.95

## Repository Structure

```
nighttime-lights-india/
├── notebook.ipynb              # Full analysis walkthrough (start here)
├── ntl_analyze.py              # Reusable analysis library
├── correlate_official_data.py  # GSDP/power correlation analysis
├── data/
│   ├── raster/                 # Pre-downloaded VIIRS GeoTIFFs (Srinagar, Leh, Manali, Ayodhya)
│   └── economic/               # Official J&K/Ladakh economic data
├── output/                     # Generated charts, CSVs, and analysis reports
└── blog/                       # Blog post draft
```

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib rasterio shapely earthengine-api requests
```

### 2. Run the notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook works with **pre-downloaded data** — no Google Earth Engine authentication needed to run the analysis and generate all charts. The GEE download step is clearly marked as optional.

### 3. (Optional) Download fresh data via GEE

If you want to download data for new regions or update existing data:

```bash
# Authenticate with Google Earth Engine
earthengine authenticate

# Run the download
python ntl_analyze.py --regions srinagar leh manali ayodhya --download --periods 2014-2026
```

You will need a [Google Earth Engine](https://earthengine.google.com/) account and project ID.

### 4. Add your own region

See Section 7 of the notebook, or add an entry to the `REGIONS` dictionary in `ntl_analyze.py`:

```python
REGIONS['my_city'] = {
    'bbox': [lon_west, lat_south, lon_east, lat_north],
    'label': 'My City',
    'color': '#FF5722',
    'marker': 'o',
}
```

## Data Sources

- **Nighttime lights:** [VIIRS DNB Monthly Composites](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG) (NOAA, via Google Earth Engine)
- **Economic data:** Reserve Bank of India, MOSPI, StatisticsTimes, IBEF

## Blog Post

[What Satellites Reveal About India's Most Transformed Cities](https://abhijitvaidya.substack.com/) <!-- TODO: Replace with actual post URL -->

## Credits

- **xKDR Forum** for the original paper and methodology that inspired this work
- **NOAA** for the VIIRS nighttime lights data
- **Google Earth Engine** for data access infrastructure

## License

MIT License. See [LICENSE](LICENSE).
