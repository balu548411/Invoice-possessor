#!/bin/bash

# Check if input path is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_path> [model_path]"
    echo "  input_path: Path to input file or directory containing invoice files"
    echo "  model_path: (Optional) Path to model weights file (default: models/final_model.h5)"
    exit 1
fi

# Set input path
INPUT_PATH=$1

# Set model path (use default if not provided)
if [ "$#" -ge 2 ]; then
    MODEL_PATH=$2
else
    MODEL_PATH="models/final_model.h5"
fi

# Create output directory
mkdir -p inference_results

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run inference
echo "Running inference on $INPUT_PATH..."
python src/inference.py \
  --input_path "$INPUT_PATH" \
  --output_dir inference_results \
  --model_path "$MODEL_PATH" \
  --img_height 800 \
  --img_width 800 \
  --max_text_length 512 \
  --num_fields 10

echo "Inference complete. Results saved to inference_results/"