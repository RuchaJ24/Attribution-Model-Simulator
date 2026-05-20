# Attribution Model Simulator

This repository contains an attribution model simulator for two marketing scenarios:

- **B2B SaaS**: long journeys, content-driven discovery, Email closing deals
- **Consumer Tech**: short journeys, Paid Social discovery, fast conversion

The notebook generates synthetic journey data, runs five attribution models, compares them against ground truth, and creates visualizations.

## Files

- `Untitled-1.ipynb` - Main notebook for data generation, attribution analysis, and visualization.
- `requirements.txt` - Python package dependencies.
- `data/` - Generated scenario datasets and ground truth files.
- `visualizations/` - Output charts saved by the notebook.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Run the notebook

Open `Untitled-1.ipynb` in Jupyter or VS Code and run the cells in order.

## Run the Streamlit app

If you want to run the app interface, create a Python script or use the notebook code as a Streamlit app.

```bash
streamlit run Untitled-1.ipynb
```

> Note: Running a `.ipynb` directly with Streamlit may require `streamlit` support for notebooks or converting it to a `.py` file.

## Project notes

- Uses synthetic data generation grounded in benchmark channel behaviors.
- Compares five attribution models: First-Touch, Last-Touch, Time-Decay, Position-Based, Shapley Value.
- Calculates MAPE against ground truth.
- Produces visual summaries for scenario comparison, model accuracy, email attribution drama, and channel position distribution.
