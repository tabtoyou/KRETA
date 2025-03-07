# Python command settings
PYTHON := python
VENV := .venv

ifeq ($(OS),Windows_NT)
	PYTHON_PATH := $(VENV)\Scripts\python
	ACTIVATE := $(VENV)\Scripts\activate.bat &&
else
	PYTHON_PATH := $(VENV)/bin/python
	ACTIVATE := . $(VENV)/bin/activate &&
endif

.PHONY: help setup filter generate evaluate

# Default help
help:
	@echo "Available commands:"
	@echo "  make setup      : Create virtual environment and install dependencies"
	@echo "  make filter     : Run image filtering (options: -d)"
	@echo "  make generate   : Generate QA data (options: -d, -r, -s)"
	@echo "  make evaluate   : Run evaluation tool"
	@echo ""
	@echo "Options:"
	@echo "  -d, --input_directory  : Input image directory path (default: ./data/images)"
	@echo "  -r, --output_directory : Result save directory path (default: ./results)"
	@echo "  -s, --save_batch      : Set batch size (default: 60)"

# Virtual environment setup
setup:
	@echo "Setting up virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@$(ACTIVATE) $(PYTHON_PATH) -m pip install --upgrade pip
	@$(ACTIVATE) $(PYTHON_PATH) -m pip install -r requirements.txt
	@echo "Setup complete!"

gpu: # not working now...
ifeq ($(OS),Windows_NT)
	@where nvidia-smi > nul 2>&1 && ( \
		echo "NVIDIA GPU detected. Installing CUDA support..." && \
		$(ACTIVATE) $(PYTHON_PATH) -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ \
	) || echo "No NVIDIA GPU detected. Skipping PaddlePaddle-GPU installation."
else
	@which nvidia-smi > /dev/null 2>&1 && ( \
		echo "NVIDIA GPU detected. Installing CUDA support..." && \
		$(ACTIVATE) $(PYTHON_PATH) -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ \
	) || echo "No NVIDIA GPU detected. Skipping PaddlePaddle-GPU installation."
endif

# Image filtering
filter:
	@echo "Running image filtering..."
	@$(ACTIVATE) $(PYTHON_PATH) filter.py $(filter-out $@,$(MAKECMDGOALS))

# QA data generation
generate:
	@echo "Generating QA data..."
	@$(ACTIVATE) $(PYTHON_PATH) main.py $(filter-out $@,$(MAKECMDGOALS))

# Run evaluation tool
evaluate:
	@echo "Running evaluation tool..."
	@$(ACTIVATE) $(PYTHON_PATH) -m streamlit run eval.py $(filter-out $@,$(MAKECMDGOALS))

%:
	@:
