FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ build-essential libffi-dev && rm -rf /var/lib/apt/lists/*

# Non-root user: this container places trades — running as root inside it
# is an unnecessary privilege-escalation surface if the container is ever
# compromised via a dependency vulnerability.
RUN useradd --create-home --shell /bin/bash botuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY app.py .
# NOTE: .env is intentionally NOT copied into the image. Baking any .env
# (even .env.example) into an image is bad practice for two reasons: (1) it
# was previously referencing a file that didn't exist in this repo at all,
# which made `docker build` fail outright; (2) even if it existed, secrets
# belong in a runtime volume mount or environment variables, never in an
# image layer that could end up in a registry. Provide real config via
# docker-compose.yml's volume mount (default) or `docker run --env-file`.
RUN mkdir -p logs && chown -R botuser:botuser /app
USER botuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
