# S&P 500 Relative Returns Data Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17971090.svg)](https://doi.org/10.5281/zenodo.17971090)

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
- **API Key**: FMP API key

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

1. Sign up at [financialmodelingprep.com](https://site.financialmodelingprep.com/register) to get your free API key
2. Open `sp500_data_pipeline.py` [line: 177] and replace `ADD_YOUR_FMP_API_KEY` with your actual key:

   ```python
   FMP_KEY = "ADD_YOUR_FMP_API_KEY"
   ```

---

## Running the Script

Make sure you are in the project directory and your environment is activated.

**MacOS / Linux / Windows:**
```bash
python sp500_data_pipeline.py
```

---

## Output

The script creates the following directories and files:

```
sp500-relative-returns/
├── data/
│   ├── fmp_stocks/           # [Generated] Individual CSV files per ticker (494 files)
│   │   └── Ticker_1.csv
│   │   └── Ticker_2.csv
│   │   └── ...
│   └── sp500_final_494_tickers_fmp.csv 
├── output/
│   └── table1_sectors.txt    # [Generated] Sector summary table
```

---

## Notes

- The script uses the FMP (FinancialModelingPrep) API for price and sector data
- API rate limits may apply depending on your FMP subscription tier
- If you encounter API errors, wait a few minutes and re-run the script

## Citation

If you use this code, please cite both the original paper and this implementation:

**BibTeX:**
```bibtex
@article{htun_forecasting_2024,
    title = {Forecasting relative returns for {S}\&{P} 500 stocks using machine learning},
    volume = {10},
    issn = {2199-4730},
    doi = {10.1186/s40854-024-00644-0},
    journal = {Financial Innovation},
    author = {Htun, Htet Htet and Biehl, Michael and Petkov, Nicolai},
    year = {2024},
    pages = {118},
}

@software{takelait_sp500_pipeline_2025,
    title = {{S\&P} 500 Relative Returns Data Pipeline},
    author = {Takelait, Fouzi},
    year = {2025},
    url = {https://github.com/ftakelait/sp500-relative-returns},
    version = {1.0.0},
}
```

---

## License

MIT License © 2025 Fouzi Takelait

---

## Contact

Fouzi Takelait — [ftakelait@gmail.com](mailto:ftakelait@gmail.com)
