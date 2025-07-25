# AchromatCFW

Utilities for analysing chromatic focal shift data exported from Zemax OpticStudio. The package
provides tools for loading spectral data, computing colour fringe width metrics and running small
experiments in notebooks.

## Setup

Create the Conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate thesis
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
│   ├── cfw_demo.ipynb                 <- A demo for the core toolbox evaluating CFW
│   ├── chl_conrady_predict.ipynb      <- Predict CHL curve and give real-time CFW
│   └── conrady_fit_validate.ipynb     <- Validate the accuracy of Conrady and ACF fit model
├── reports/            <- Generated analysis results
├── references/
│   └── schott-optical-glass.xlsx
├── src/
│   └── achromatcfw/
│       ├── core/       <- JIT accelerated CFW routines
│       ├── io/         <- Data loading utilities
│       ├── data/       <- Spectral CSV files packaged with the code
│       └── zemax_utils.py
└── tests/              <- Pytest unit tests
```

## Usage

The Notebooks can be run directly either with given file in database or the predicted 
CHL data. The Zemax helper is not finished yet.
