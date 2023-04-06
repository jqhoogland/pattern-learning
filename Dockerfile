FROM ubuntu:latest

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y python3.10 python3-pip && \
    python3 -m pip install --upgrade pip

# Copy the local module directory into the Docker image
COPY dominoes /app/dominoes

# Copy the requirements.txt file into the Docker image
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Set the working directory to the app directory
WORKDIR /app

ARG WANDB_SWEEP_ID
ENV WANDB_SWEEP_ID=$WANDB_SWEEP_ID

# Run your Python script
CMD ["python3", "sweep.py "]