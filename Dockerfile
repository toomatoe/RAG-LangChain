FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home appuser
COPY app.py ./
COPY data/books ./data/books
RUN mkdir -p data/chroma_db && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
