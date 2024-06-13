# Use an official Python runtime as a parent image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["python", "app.py"]