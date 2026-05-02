FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY claudev.py .
COPY products.txt .

CMD ["uvicorn", "claudev:app", "--host", "0.0.0.0", "--port", "8000"]
 
