# Predictive Maintenance System

A Flask web application that predicts industrial machine failures within the next seven days using a trained Random Forest model and the provided CSV dataset.

## Features
- Professional landing page and dashboard design
- Machine learning model information page with metrics and charts
- Prediction page for manual machine parameter input
- Clear risk categories and maintenance recommendations
- PDF report generation with prediction summary

## Setup
1. Install Python 3.11 or newer.
2. Open the project folder in a terminal.
3. Install dependencies:
   ```bash
   py -m pip install -r requirements.txt
   ```
4. Train the model:
   ```bash
   py train_model.py
   ```
5. Run the application:
   ```bash
   py app.py
   ```
6. Open the browser at `http://127.0.0.1:5000`.

## Notes
- The model is trained from `accepted_datasets_cleaned.csv`.
- Generated artifacts include `model.pkl`, `preprocessor.pkl`, and `model_info.json`.
- No database or IoT sensors are used.
