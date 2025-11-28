# Menggunakan base image Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . /app

# Install dependencies jika ada
RUN pip install -r requirements.txt || true

# Jalankan script python sebagai default
CMD ["python", "hello.py"]
