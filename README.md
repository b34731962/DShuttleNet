# DShuttleNet: Badminton Match Analytics

## Project Structure

```
DShuttleNet/
├── main.ipynb                    # Main orchestration notebook
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── data/                         # Raw and processed datasets
│   ├── tournament.csv           # Tournament metadata
│   ├── match.csv                # Match details (event, duration, sets)
│   ├── set.csv                  # Set information
│   ├── rally.csv                # Rally-level data
│   ├── shot.csv                 # Individual shots with coordinates & types
│   ├── player.csv               # Player information
│   └── merge_data.csv           # Output: merged data with engineered features
│
├── src/
│   ├── data_pipeline/           # Data processing and feature engineering
│   │   ├── clean.py            # Data cleaning (outlier removal)
│   │   └── feature_engineering.py  # Tactical feature extraction
│   │
│   ├── analysis/                # Multi-dimensional analysis modules
│   │   ├── general_analyze.py   # General shot statistics & 3D analysis
│   │   ├── defensive_failure.py # Defensive breakdown patterns
│   │   ├── matches_analyze.py   # Match intensity comparison
│   │   ├── ccei_analysis.py     # Court Coverage & Formation analysis
│   │   ├── shottype_heatmap.py  # Position density heatmaps (KDE)
│   │   ├── shottype_pos.py      # Court occupancy visualization
│   │   ├── tlai_analysis.py     # Threat Level Awareness Index
│   │   └── old_general_analyze_copy.py  # Legacy analysis
│   │
│   └── utils/                   # Utility functions
│       └── data_utils.py        # Data filtering and coordinate processing
│
└── img/                          # Generated visualizations
    ├── 3dplot/                  # 3D scatter plots of shot analysis
    ├── angle_grouped/           # Bar charts of shot angles
    └── Coordination Analysis/   # Formation and positioning heatmaps
```

## Usage

### Prerequisites
```bash
python 3.11.15
# Install dependencies:
pip install -r requirements.txt
```

### Running the Analysis
1. **Place raw data** in `data/` directory with expected CSV files
2. **Open and run** `main.ipynb` in Jupyter:
   ```bash
   jupyter notebook main.ipynb
   ```
3. The notebook will:
   - Load and merge all data tables
   - Execute data cleaning
   - Engineer tactical features
   - Generate `merge_data.csv`
   - Run all analysis modules
   - Output visualizations to `img/`

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 3.0.2 | Data manipulation and analysis |
| numpy | 2.4.4 | Numerical computing |
| matplotlib | 3.10.8 | 2D/3D plotting and visualization |
| seaborn | 0.13.2 | Statistical data visualization |
| scipy | 1.17.1 | Scientific computing (KDE, statistics) |

## Data Format

### Engineered Features (merge_data.csv)
- Hitting_Zone (1-9, 10-16 for out-of-bounds)
- Angle_Bin (0-5, where 5 = error)
- ball_type (mapped categories)
- Formation (Attack/Defense/Transition)
- Defender response metrics

### Coordinate System
- Court dimensions: 61m × 134m (badminton doubles court)
- Player tracking with (X, Y) coordinates
- 4-player tracking (A, B, C, D) per shot

## License

[Add appropriate license information]

## Acknowledgments

This project analyzes badminton match data to provide tactical insights for players, coaches, and match analysts.

---