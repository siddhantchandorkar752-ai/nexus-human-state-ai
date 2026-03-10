FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1 libxcb1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . /app
EXPOSE 7860
CMD ["python", "app.py"]
