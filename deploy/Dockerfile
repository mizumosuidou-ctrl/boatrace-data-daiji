FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-minamo.txt .
RUN pip install --no-cache-dir -r requirements-minamo.txt
COPY minamo ./minamo
COPY web ./web
CMD ["python", "-m", "minamo", "run"]
