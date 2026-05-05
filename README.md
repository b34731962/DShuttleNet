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
│   │   └── tlai_analysis.py     # Threat Level Awareness Index
│   │   
│   │
│   └── utils/                   # Utility functions
│       └── data_utils.py        # Data filtering and coordinate processing
│
└── img/                          # Generated visualizations
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

## License

[Add appropriate license information]

## Acknowledgments

This project analyzes badminton match data to provide tactical insights for players, coaches, and match analysis.

---
