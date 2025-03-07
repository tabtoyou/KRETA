# Python command settings
PYTHON := python
VENV := .venv

# Default values
GPU ?= 0
INPUT_DIR ?= ./data/images
OUTPUT_DIR ?= ./results
SAVE_BATCH ?= 60

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
	@echo "  make setup              : Create virtual environment and install dependencies"
	@echo "  make filter            : Run image filtering"
	@echo "  make generate          : Generate QA data"
	@echo "  make evaluate          : Run evaluation tool"
	@echo ""
	@echo "Options (set as environment variables):"
	@echo "  GPU=1                  : Enable GPU support (default: 0)"
	@echo "  INPUT_DIR=path         : Input image directory path (default: ./data/images)"
	@echo "  OUTPUT_DIR=path        : Result save directory path (default: ./results)"
	@echo "  SAVE_BATCH=number      : Set batch size (default: 60)"

# Virtual environment setup
setup:
	@echo "Setting up virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@$(ACTIVATE) $(PYTHON_PATH) -m pip install --upgrade pip
	@$(ACTIVATE) $(PYTHON_PATH) -m pip install -r requirements.txt
	@if [ "$(GPU)" = "1" ]; then \
		echo "Installing with GPU support..."; \
		$(ACTIVATE) $(PYTHON_PATH) -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/; \
	else \
		echo "Installing CPU version..."; \
		$(ACTIVATE) $(PYTHON_PATH) -m pip install paddlepaddle; \
	fi
	@echo "Setup complete!"

# Image filtering
filter:
	@echo "Running image filtering..."
	@$(ACTIVATE) $(PYTHON_PATH) filter.py --input_directory $(INPUT_DIR)

# QA data generation
generate:
	@echo "Generating QA data..."
	@$(ACTIVATE) $(PYTHON_PATH) main.py --input_directory $(INPUT_DIR) --output_directory $(OUTPUT_DIR) --save_batch $(SAVE_BATCH)

# Run evaluation tool
evaluate:
	@echo "Running evaluation tool..."
	@$(ACTIVATE) $(PYTHON_PATH) -m streamlit run eval.py

%:
	@:
