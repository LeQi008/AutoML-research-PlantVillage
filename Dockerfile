
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

#==== System dependencies ====
RUN apt-get update && apt-get install -y --no-install-recommends bash git \
    && rm -rf /var/lib/apt/lists/*

#==== Python dependencies ====
COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install -r requirements.txt

#==== Application source ====
COPY . .

RUN mkdir -p data experiments literature \
    && chmod +x scripts/run_experiment.sh

CMD ["python", "agent/agent.py"]
