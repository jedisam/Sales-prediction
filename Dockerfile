FROM python:3.9-slim-buster
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN apt-get update -y
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["python", "app.py"]
