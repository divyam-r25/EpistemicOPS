FROM python:3.11-slim
WORKDIR /app
RUN pip install httpx fastapi uvicorn pydantic
COPY ./mock_apis/drift_injector/main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
