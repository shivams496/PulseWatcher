FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit torch numpy pandas \
    matplotlib seaborn scikit-learn plotly wfdb

COPY . .

EXPOSE 7860

CMD ["/bin/bash", "-c", "python -m streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true"]