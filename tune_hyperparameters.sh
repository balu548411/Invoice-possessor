#!/bin/bash

# Create output directory
mkdir -p tuning_results

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run hyperparameter tuning
echo "Starting hyperparameter tuning..."
python src/hyperparameter_tuning.py \
  --image_dir training_data/images \
  --label_dir training_data/labels \
  --output_dir tuning_results \
  --img_height 800 \
  --img_width 800 \
  --max_text_length 512 \
  --num_fields 10 \
  --train_split 0.8 \
  --tuning_epochs 3 \
  --fast_mode \
  --max_train_samples 50 \
  --max_val_samples 20

echo "Hyperparameter tuning complete! Results saved to tuning_results/" 