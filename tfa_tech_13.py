# BULL_VLM - Image captioning service for NGSI-LD notifications using Florence 2
# Copyright (C) 2026 Bull SAS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
 
# Metadata as described in the documentation
LABEL maintainer="Advanced Computing & AI. Strategy and Innovation" \
      version="1.0" \
      description="Image with CUDA 12.1.0, pytorch, llava, and other dependencies."
 
# Arguments to build Docker Image using CUDA
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA True
 
ENV CUDA_HOME /usr/local/cuda/
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME /home/bull/app/huggingface

# Install essential packages and Jupyter Notebook
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    python3-opencv \
    wget \
    g++ && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch, torchvision, and torchaudio with CUDA support
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#RUN pip install torch

# Install other dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y git
 
RUN pip install --upgrade pip

WORKDIR /home/bull

# Copy the necessary files into the working directory (/app) in the container
# This includes all files from the current directory (where the Dockerfile is located) into the /app directory within the container
COPY . /home/bull/app

# Set the working directory to /app. This directory will be the default location for any subsequent commands executed within the container
WORKDIR /home/bull/app

# The port on which the application will listen for incoming connections
ENV PORT=8003
# The basepath environment variable
ENV BASE_PATH='/vlm'

# The URL of the message broker to connect to
ENV BROKER_URL="https://orion.tema.digital-enabler.eng.it"
# The ID of the topic to subscribe to or publish messages to
ENV BROKER_ENTITY_ID=topic123
# The type of the entity (topic or queue) to interact with
ENV BROKER_TYPE_ID="ImageVQA"
# The ID of the subscription to use for receiving messages
ENV BROKER_SUBSCRIPTION_ID=subscription123
# The URL to which callback notifications will be sent
ENV CALLBACK_URL=http://callback.example.com

# Whether debugging mode is enabled. Default is set to False
ENV DEBUG=False        
# The processing unit to be used (cpu or gpu). Default is set to cpu
ENV PROCESSING_UNIT=gpu


# Install the dependencies specified in requirements.txt
# This command installs all Python packages listed in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade transformers==4.49.0


# Run the application when the container starts
# This command specifies the default command to execute when the container is launched
# It runs the Python script named "app.py", which is the main entry point for the application
CMD ["python3", "tfa_tech_13.py"]