# ☁️ weBIGeo Cloud Server

![Version](https://img.shields.io/badge/version-1.1-blue) ![License](https://img.shields.io/github/license/webigeo/clouds-server)

Preprocessing server for real-time volumetric cloud rendering in the browser. Developed by **Wendelin Muth** as part of his bachelor thesis at TU Wien.

> [Real-Time Volumetric Rendering of Meteorological Cloud Data](https://www.cg.tuwien.ac.at/research/publications/2026/muth-2026-clouds/)
> Wendelin Muth, Research Unit of Computer Graphics, TU Wien, March 2026
> Supervisor: Manuela Waldner

This server is built for the [weBIGeo](https://github.com/weBIGeo/webigeo) web-based geographic visualization project. It fetches weather forecast data from DWD (ICON-D2 model), processes it into compressed tile hierarchies, and serves them for ray-marching cloud rendering.

## Setup

### using CONDA

> **Note:** I ran into issues installing `cfgrib` via pip on Windows because it depends on the native ecCodes C library, which pip didnt seem to properly provide on Windows (at least not for a venv). Installing `eccodes` and `cfgrib` through conda-forge (as above) resolved this.

```bash
conda create -n clouds python=3.11
conda activate clouds
conda install -c conda-forge eccodes cfgrib
pip install -r requirements.txt
```

### no virtual environment

```bash
pip install -r requirements.txt
```

## Usage

```bash
python server.py
```

Configuration is managed and documented in [`config.py`](config.py). See [`API.md`](API.md) for the full API-Endpoint reference.
