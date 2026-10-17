
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

COPY app.py .
COPY train_model.py .
COPY model.pkl .
COPY templates/ templates/
COPY static/ static/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
