FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application

COPY langgraph_fastapi.py .

EXPOSE 8000

# Run the API server
CMD ["python", "langgraph_fastapi.py"]
