FROM tensorflow/tensorflow:latest-gpu-py3

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3.7-dev \
    python3-pip \
    python3-setuptools && apt-get clean

RUN python3 -m pip install --upgrade pip && python3 -m pip install PyYAML scipy scikit-learn tqdm

WORKDIR /app
ADD . /app

VOLUME /app/datasets
VOLUME /app/logs
VOLUME /app/training_checkpoints

CMD ["python3", "main.py"]
