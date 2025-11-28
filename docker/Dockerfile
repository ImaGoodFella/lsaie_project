FROM nvcr.io/nvidia/pytorch:25.06-py3

# setup
RUN apt-get update && apt-get install python3-pip python3-venv -y
	
# Install the rest of dependencies.
RUN pip install \
	datasets \
	transformers \
	accelerate \
	wandb \
	dacite \
	pyyaml \
	numpy \ 
	packaging \
	safetensors \
	tqdm \
	sentencepiece \
	tensorboard \
	pandas \
	jupyter \
	deepspeed \
	seaborn \
	deepspeed \
	bitsandbytes

# Create a work directory
RUN mkdir -p /workspace
