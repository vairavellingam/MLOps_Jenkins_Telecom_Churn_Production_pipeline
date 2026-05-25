FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e .


RUN python training_pipeline.py

EXPOSE 5000

ENV FLASK_APP=application.py

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "application:app"]
