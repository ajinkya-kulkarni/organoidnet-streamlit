# OrganoIDNet Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segmentorganoids.streamlit.app/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Interactive organoid segmentation and analysis for Pancreatic Ductal Adenocarcinoma (PDAC) research.

- **Segmentation models**: CellSeg-PyTorch and Cellpose (original)
- **Analysis**: Live/dead classification, morphology distributions, size categories
- **Output**: Instance masks (TIFF), per-organoid stats (CSV)

## Usage

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or deploy directly on [Streamlit Cloud](https://streamlit.io/cloud).

## Publication

- OrganoIDNet paper: https://doi.org/10.1007/s13402-024-00958-2
- Training code & dataset: https://github.com/ajinkya-kulkarni/PyOrganoIDNet
