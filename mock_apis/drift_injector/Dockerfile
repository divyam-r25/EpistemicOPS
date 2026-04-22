FROM python:3.11-slim
WORKDIR /app
RUN pip install httpx
COPY ./mock_apis/drift_injector/main.py .
CMD ["python", "main.py"]
