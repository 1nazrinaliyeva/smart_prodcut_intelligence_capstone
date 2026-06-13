FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY src ./src
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir .

EXPOSE 8501

ENTRYPOINT ["smart-product-intelligence", "--host", "0.0.0.0"]
CMD ["--port", "8501"]
