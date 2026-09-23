import os
import io
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.preprocessing import RobustScaler

# Import model definition module so PyTorch can unpickle the model object
import model

LSTM_PATH = "./models/lstm_non-stationary.pt"
LR_PATH = "./models/linear_regressor_non-stationary.pkl"

def load_saved_models():
    models = {}

    if os.path.exists(LSTM_PATH):
        try:
            lstm_model = torch.load(LSTM_PATH, map_location=torch.device('cpu'), weights_only=False)
            if hasattr(lstm_model, 'eval'):
                lstm_model.eval()
            models['LSTM (Non-Stationary)'] = lstm_model
            print("Loaded LSTM model.")
        except Exception as e:
            print(f"Error loading LSTM: {e}")

    if os.path.exists(LR_PATH):
        try:
            with open(LR_PATH, 'rb') as f:
                lr_model = pickle.load(f)
            models['Linear Regression (Non-Stationary)'] = lr_model
            print("Loaded Linear Regression model.")
        except Exception as e:
            print(f"Error loading Linear Regression: {e}")

    return models

LOADED_MODELS = load_saved_models()


def predict_next_day_close(csv_text, model_choice):
    if not csv_text or not csv_text.strip():
        return "Error: Please enter CSV data.", None

    if model_choice not in LOADED_MODELS:
        return f"Error: Selected model '{model_choice}' not found.", None

    try:
        # Parse CSV input
        df = pd.read_csv(io.StringIO(csv_text.strip()))
        df.columns = df.columns.str.strip().str.capitalize()

        if 'Close' not in df.columns:
            return "Error: Input CSV must contain at least a 'Close' price column.", None

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            dates = list(df['Date'])
        else:
            dates = list(pd.date_range(start='2026-09-01', periods=len(df), freq='D'))

        # Scale Close prices (X_t)
        close_values = df[['Close']].values
        scaler = RobustScaler()
        scaled_close = scaler.fit_transform(close_values)

        selected_model = LOADED_MODELS[model_choice]
        
        # Determine model input feature expectations
        if "LSTM" in model_choice:
            expected_features = getattr(selected_model.lstm, 'input_size', 16) if hasattr(selected_model, 'lstm') else 16
        else:
            expected_features = getattr(selected_model, 'n_features_in_', 1)

        predictions = []
        

        for i in range(len(scaled_close)):
            feat = scaled_close[i:i+1] # Input at time t
            if feat.shape[1] != expected_features:
                feat = np.repeat(feat, expected_features, axis=1)

            if "LSTM" in model_choice:
                x_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred_scaled = selected_model(x_tensor).numpy().flatten()[0]
            else:
                pred_scaled = selected_model.predict(feat).flatten()[0]

            pred_y = scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred_y)

  
        # 1. Historical Predictions vs Actuals (Day 2 to Day N)
        hist_actual_dates = dates[1:]
        hist_actual_prices = df['Close'].values[1:]
        hist_predictions = predictions[:-1]  # Predictions made on Days 1..(N-1) for Days 2..N

        # 2. Next Unseen Day Prediction (Day N+1)
        next_unseen_date = dates[-1] + pd.Timedelta(days=1)
        next_day_forecast = predictions[-1]  # Forecast made on latest input for tomorrow

        # Combine dates and predictions for full plot
        plot_pred_dates = hist_actual_dates + [next_unseen_date]
        plot_pred_values = hist_predictions + [next_day_forecast]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        # Historical Actual Line
        ax.plot(dates, df['Close'], label='Actual Historic Close (X_t)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
        
        # Shifted Forecast Line (y_t+1)
        ax.plot(plot_pred_dates, plot_pred_values, label=f'Model Forecast (y_t+1) — {model_choice}', color='#ff7f0e', linestyle='--', linewidth=2, marker='x', markersize=4)

        # Highlight tomorrow's standalone forecast point
        ax.scatter([next_unseen_date], [next_day_forecast], color='#d62728', s=120, zorder=5, label=f'Next Day Forecast (${next_day_forecast:.2f})')

        ax.set_title("Input Price (X_t) vs Next-Day Target Forecast (y_t+1)", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left')
        plt.tight_layout()

        status = (
            f"✅ Forecast complete using {model_choice}!\n\n"
            f"• Latest Input Price ({dates[-1].strftime('%Y-%m-%d')}): ${df['Close'].iloc[-1]:.2f}\n"
            f"• Predicted Next-Day Price ({next_unseen_date.strftime('%Y-%m-%d')}): ${next_day_forecast:.2f}"
        )

        return status, fig

    except Exception as e:
        return f"Prediction Error: {str(e)}", None

# Sample CSV
DEFAULT_CSV = """Date,Close
2026-09-01,103.20
2026-09-02,105.00
2026-09-03,107.40
2026-09-04,104.10
2026-09-05,105.80
"""

# --------------------------------------------------------------------
# 3. Gradio Interface Layout
# --------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Target y Predictor") as demo:
    gr.Markdown("# 📈 Next-Day Stock Closing Price ($y_{t+1}$) Predictor")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(LOADED_MODELS.keys()) if LOADED_MODELS else ["No Models Found"],
                value=list(LOADED_MODELS.keys())[0] if LOADED_MODELS else None,
                label="Select Trained Model"
            )

            csv_input = gr.Textbox(
                value=DEFAULT_CSV,
                lines=8,
                label="Historical Input Prices (X_t)"
            )

            predict_btn = gr.Button("Predict Tomorrow's Price (y_t+1)", variant="primary")

        with gr.Column(scale=2):
            output_status = gr.Textbox(label="Forecast Result", lines=4, interactive=False)
            output_graph = gr.Plot()

    gr.Markdown("---")
    gr.Markdown("### Research Notes & Findings: notion_link")

    predict_btn.click(
        fn=predict_next_day_close,
        inputs=[csv_input, model_selector],
        outputs=[output_status, output_graph]
    )

if __name__ == "__main__":
    demo.launch()