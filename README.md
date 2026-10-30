Copyright © 2026 Bull® SAS. All rights reserved

# BULL_VLM

## Overview
This repository provides a Docker container image based on `nvidia/cuda:12.1.0-devel-ubuntu22.04`. The container is pre-configured with CUDA 12.1.0, PyTorch, LLAVA, and essential dependencies for running machine learning and object tracking tasks. It uses Florence 2 to generate image captions.

## Features
- **Base Image**: NVIDIA CUDA 12.1.0 development environment
- **ML Frameworks**: PyTorch, torchvision, and torchaudio with CUDA support
- **Pre-installed Tools**: Florence 2
- **Custom Scripts**: Includes `tfa_tech_13.py` as the main application script
- **Environment Variables**: Flexible configuration for runtime parameters

## Prerequisites
- NVIDIA GPU with compatible drivers
- Docker installed on the host machine

## Get the Image from the registry
```bash
docker pull ghcr.io/he-tema/atos_vlm_image:1.0
```

## Run the Container
Run the container with the necessary environment variables:

```bash
docker run --rm \
  -e PORT=8003 \
  -e BASE_PATH='/vlm' \
  -e BROKER_URL='https://orion.tema.digital-enabler.eng.it' \
  -e BROKER_TYPE_ID='ImageVQA' \
  -e DEBUG=false \
  -e PROCESSING_UNIT=gpu \
  -e endpoint=<endpoint> \
  -e access_key=<access-key> \
  -e secret_key=<secret-key> \
  -p 8003:8003 \
  --gpus=all \
  atos_vlm_image
```

### Environment Variables
#### Mandatory
- `PORT`: The port on which the application listens
- `BASE_PATH`: Base path for API routes
- `BROKER_URL`: URL of the message broker
- `BROKER_TYPE_ID`: Entity type for interaction
- `endpoint`: URL of the S3-compatible object storage service.
- `access_key`: Access key used to authenticate with the object storage service.
- `secret_key`: Secret key associated with the access key, used for authentication with the object storage service.

#### Optional
- `DEBUG`: Enable debugging mode (`True` or `False`, default: `False`)
- `PROCESSING_UNIT`: Specify the processing unit (`cpu` or `gpu`, default: `gpu`)

### Application Entry Point
The application is launched using the following command:

```bash
CMD ["python3", "tfa_tech_13.py"]
```


## Maintainer
For any questions or issues, please contact: Advanced Computing & AI. Strategy and Innovation

