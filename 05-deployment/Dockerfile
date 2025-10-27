FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install fastapi==0.120.1 scikit-learn==1.6.1 uvicorn==0.38.0

# Copy model and app
COPY pipeline_v1.bin /app/pipeline_v1.bin
COPY app.py /app/app.py

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]