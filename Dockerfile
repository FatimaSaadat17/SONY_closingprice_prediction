FROM python:3.10-slim

WORKDIR ./
COPY . .
RUN pip install --no-cache-dir gradio
RUN pip install --upgrade pip
RUN pip install --default-timeout=900 torch
RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install pytorch-lightning
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
