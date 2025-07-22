# AchromatCFW

Utilities for analysing chromatic focal shift data exported from Zemax OpticStudio. The package
provides tools for loading spectral data, computing colour fringe width metrics and running small
experiments in notebooks.

## Setup

Create the Conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate masterthesis
```

Alternatively install the minimal requirements with pip:

```bash
pip install -r requirements.txt
```

## Running the tests

```bash
pytest -q
```

## Repository layout

```
├── LICENSE             <- Project license
├── README.md           <- This file
├── environment.yml     <- Conda environment description
├── requirements.txt    <- Minimal pip requirements
├── notebooks/          <- Example notebooks
├── reports/            <- Generated analysis results
├── src/
│   └── achromatcfw/
│       ├── core/       <- JIT accelerated CFW routines
│       ├── io/         <- Data loading utilities
│       ├── data/       <- Spectral CSV files packaged with the code
│       └── zemax_utils.py
└── tests/              <- Pytest unit tests
```

## Usage

The Zemax helper can be run directly to fetch the chromatic focal shift curve and
compute fringe metrics:

```bash
python -m achromatcfw.zemax_utils path/to/system.zmx \
    --defocus-range 500 --xrange 200 --F 8.0 --gamma 1.0
```

The script prints the maximum and mean colour fringe width for the specified
defocus range. Use `-h` to see all available options.
