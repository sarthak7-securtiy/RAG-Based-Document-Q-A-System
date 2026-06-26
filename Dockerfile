# Use multi-stage build to keep the image size small
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for PyMuPDF and Chroma
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
