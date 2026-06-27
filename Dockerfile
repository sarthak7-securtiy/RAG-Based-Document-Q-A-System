FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyMuPDF and Chroma
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p data/uploaded_docs chroma_db

# Streamlit config for headless deployment
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Cloud Run uses PORT env var (single port)
# We expose 8501 as the primary (Streamlit) port
EXPOSE 8501

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
