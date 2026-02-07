FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
		gcc \
		g++ \
		python3-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt
CMD ["python3","app.py"]