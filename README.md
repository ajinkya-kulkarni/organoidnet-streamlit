# OrganoIDNet Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segmentorganoids.streamlit.app/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Interactive organoid segmentation and analysis for Pancreatic Ductal Adenocarcinoma (PDAC) research.

- **Segmentation model**: CellSeg-PyTorch
- **Analysis**: Live/dead classification, morphology distributions, size categories
- **Output**: Instance masks (TIFF), per-organoid stats (CSV)

## Usage

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Or deploy directly on [Streamlit Cloud](https://streamlit.io/cloud).

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run mypy streamlit_app.py patch_inference.py
```

## Publications

- **OrganoIDNet model**: [Kulkarni et al. (2024)](https://link.springer.com/article/10.1007/s13402-024-00958-2)
- **PDAC organoid dataset**: [Kulkarni et al. (2024)](https://www.nature.com/articles/s41597-024-03631-3)
