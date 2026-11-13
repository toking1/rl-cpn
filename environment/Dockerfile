FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements-py313.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /workspace/code
