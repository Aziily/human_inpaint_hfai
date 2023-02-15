export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="data/densepose/masks"
export OUTPUT_DIR="outputs/test_node_1"

echo "[log] Start training"
hfai python \
  run.py \
  train_human_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a coco person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20000 \
  -- \
  --name "densepose_node_1" \
  --nodes 1 \
  --priority 10 
echo "[log] starting"

# hfai logs -f test