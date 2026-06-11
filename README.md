# Medical Cost Prediction

A compact project to explore and predict individual medical insurance costs using a public US insurance dataset. Includes an analysis notebook and a Streamlit dashboard for quick experiments.

## Features
- Exploratory data analysis and preprocessing
- Train and save simple regression models (Ridge, Random Forest)
- Interactive Streamlit dashboard for visualization and prediction
- Includes the analysis notebook

## How to run
Create and activate a local virtual environment, install dependencies, then run the dashboard:

```bash
python3 -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
streamlit run app.py
```

## Model information
The Models I used `scikit-learn`. My project target is to demonstrates Ridge regression and Random Forest; the notebook documents training, evaluation and simple feature engineering.

Key inputs: age, sex, bmi, children, smoker, region (plus a few engineered features).

## Dashboard pages
- Home: project overview
- Data Exploration: interactive charts
- Prediction: enter values to get an estimate
- Model Info: metrics and notes

## Files
- `app.py` — Streamlit app
- `medical_cost_prediction.ipynb` — analysis notebook
- `insurance.csv` — dataset
- `models/insurance_prediction_model.pkl` — saved model artifacts
- `requirements.txt` — dependencies

## Notes on data source
This uses a common US insurance dataset for demonstration. I wanted to use for East Africa for my local context but failed to get the dataset.

Developed by Mule Samuel

