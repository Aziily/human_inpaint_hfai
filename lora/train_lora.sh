export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export MODEL_PATH="/ceph-jd/pub/jupyter/chenhao/notebooks/huggingface/diffusers/models--runwayml--stable-diffusion-inpainting/snapshots/caac1048f28756b68042add4670bec6f4ae314f8"
export INSTANCE_DIR="data/densepose/masks"
export NAME="train_lora_sparse"
export OUTPUT_DIR="outputs/$NAME"
export EVAL_DATA_ROOT="data/inference/images/"

echo "[log] Start training"
hfai python \
  run.py \
  $NAME.py \
  --name=$NAME\
  --pretrained_model_name=$MODEL_NAME \
  --pretrained_model_path=$MODEL_PATH \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a coco person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20000 \
  --eval_step=100 \
  --eval_data_root=$EVAL_DATA_ROOT \
  --eval_num=4 \
  -- \
  --name $NAME \
  --nodes 1 \
  --priority 10 
echo "[log] starting"

# hfai logs -f test