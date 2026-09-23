# Sony Stock Price Forecasting: PyTorch LSTM vs. Linear Regression

[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF5000?logo=gradio&logoColor=white)](https://gradio.app/)
[![PyTorch Lightning](https://img.shields.io/badge/Framework-PyTorch%20Lightning-7952B3?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![Notion](https://img.shields.io/badge/Research-Notion%20Notes-000000?logo=notion&logoColor=white)](https://precious-columnist-a00.notion.site/Time-Series-Forecasting-for-SONY-Stocks-Closing-prices-3e4f4ed388e280cb8559d0cda029f77d)

An end-to-end time series forecasting and deployment project evaluating **13 years of non-stationary Sony Group Corporation (SONY)** stock market data. This project benchmarks a **PyTorch Lightning LSTM** architecture against a **Scikit-Learn Linear Regression** baseline to predict next-day closing prices ($y_{t+1}$) using engineered lag features and rolling window indicators, completely containerized with **Docker** and deployed via an interactive **Gradio GUI**.

📌 **Research & Technical Notes:** Detailed EDA, feature engineering choices, and statistical findings are documented on [Notion](https://precious-columnist-a00.notion.site/Time-Series-Forecasting-for-SONY-Stocks-Closing-prices-3e4f4ed388e280cb8559d0cda029f77d).

---

## 🚀 Quick Start (Docker Deployment)

Run the containerized application locally with zero environment configuration:

```bash
# 1. Clone the repository
git clone https://github.com/FatimaSaadat17/SONY_closingprice_prediction.git

# 2. Build the Docker image
docker build -t gradio-app .

# 3. Run the container
docker run -p 7860:7860 gradio-app

