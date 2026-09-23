# Sony Stock Price Forecasting: PyTorch LSTM vs. Linear Regression

- uses kaggle dataset: SONY Group Corporation Stock data (2014–2026) (https://www.kaggle.com/datasets/nilesh2042/sony-stock-data/data)
- GUI interface created using Gradio and containerized and deployed via docker
- predicts the next day's closing prices by extracting moving averages, lag features and so on.
- Stack
  1) LSTM was built using the Pytorch library
  2) Linear regressor was built sci-kit learn
  3) LSTM model was trained using the pytorch-lightning API
  4) Both models were packaged and deployed using Gradio + docker
  6) other packages - tsf, nixtla, pandas, numpy
- check out my research the notes for feature engineering choices and exploratory data analysis details- https://precious-columnist-a00.notion.site/Time-Series-Forecasting-for-SONY-Stocks-Closing-prices-3e4f4ed388e280cb8559d0cda029f77d



- to use the model:
  1) git clone https://github.com/FatimaSaadat17/SONY_closingprice_prediction.git
  2) docker build -t gradio-app .
  3) docker run -p 7860:7860 gradio-app
  4) then simply open http://localhost:7860/ on your default web browser to view the GUI

