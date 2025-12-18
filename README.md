# S&P 500 Relative Returns Data Pipeline

Data pipeline for replicating the S&P 500 stock forecasting methodology from:

> Htun, H. H., Biehl, M., & Petkov, N. (2024). Forecasting relative returns for S&P 500 stocks using machine learning. Financial Innovation, 10, 118.  
> PDF link: https://link.springer.com/content/pdf/10.1186/s40854-024-00644-0.pdf

---

## Features

- Fetches historical S&P 500 component data (snapshot as of 2022-01-01)
- Downloads daily price data from FinancialModelingPrep (FMP) API
- Validates data quality (1259 trading days per ticker)
- Saves sector information and ticker lists for downstream use

---

## Requirements

- **Python**: 3.10 or higher
- **OS**: Windows, macOS, or Linux
- **API Key**: FMP API key (already embedded in script, or replace with your own)

---

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n sp_forecast python=3.13 -y

# Activate the environment
conda activate sp_forecast

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using pip with venv

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## API Key Setup

This script requires a [FinancialModelingPrep (FMP)](https://financialmodelingprep.com/) API key.

1. Sign up at [financialmodelingprep.com](https://financialmodelingprep.com/) to get your free API key
2. Open `sp500_data_pipeline.py` and replace `ADD_YOUR_FMP_API_KEY` with your actual key:
   ```python
   FMP_KEY = "your_actual_api_key_here"
   ```

---

## Running the Script

Make sure you are in the project directory and your environment is activated.

**macOS / Linux:**
```bash
python sp500_data_pipeline.py
```

**Windows:**
```cmd
python sp500_data_pipeline.py
```

---

## Output

The script creates the following directories and files:

```
sp500-relative-returns/
├── data/
│   ├── fmp_stocks/           # Individual CSV files per ticker (494 files)
│   └── sp500_final_494_tickers_fmp.csv
├── output/
│   └── table1_sectors.txt    # Sector summary table
```

---

## Expected Runtime

- **First run**: ~10-15 minutes (downloading 494 ticker price histories from FMP API)
- **Subsequent runs**: ~2-3 minutes (cached data is skipped)

---

## Notes

- The script uses the FMP (FinancialModelingPrep) API for price and sector data
- API rate limits may apply depending on your FMP subscription tier
- If you encounter API errors, wait a few minutes and re-run the script

---

## License

MIT License
