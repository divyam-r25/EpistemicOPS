FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install fastapi uvicorn httpx pydantic
COPY ./mock_apis/metrics_api /app
ENV DRIFT_MODE=stable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
