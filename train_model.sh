#!/bin/bash

# Create necessary directories
mkdir -p models logs model_checkpoints model_visualization evaluation_results

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Step 1: Visualize model architecture
echo "Visualizing model architecture..."
python src/visualize_model.py

# Step 2: Train the model
echo "Starting model training..."
python src/train.py \
  --image_dir training_data/images \
  --label_dir training_data/labels \
  --model_dir models \
  --log_dir logs \
  --checkpoint_dir model_checkpoints \
  --img_height 800 \
  --img_width 800 \
  --max_text_length 512 \
  --num_fields 10 \
  --train_split 0.8 \
  --epochs 20 \
  --batch_size 8

# Step 3: Evaluate the model
echo "Evaluating trained model..."
python src/evaluate.py \
  --image_dir training_data/images \
  --label_dir training_data/labels \
  --model_path models/final_model.h5 \
  --results_dir evaluation_results

echo "Training pipeline complete!" 