FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY src ./src
COPY documents ./documents

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
