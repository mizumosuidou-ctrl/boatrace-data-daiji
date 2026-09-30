FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
WORKDIR /app
COPY requirements-minamo.txt .
RUN pip install --no-cache-dir -r requirements-minamo.txt
COPY minamo ./minamo
COPY web ./web
CMD ["python", "-m", "minamo", "run"]
