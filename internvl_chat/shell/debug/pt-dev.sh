set -x

GPUS=${GPUS:-8}


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

TRAIN_VER=$1

OUTPUT_DIR="/share/zwp/WORK_DIR/internvl_debug/pt_${TRAIN_VER}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Stage: Stage 1 (MLP Warmup)
# Architecture: InternViT-300M-448px-V2_5 + MLP + Qwen2.5-0.5B-Instruct
# Trainable Components: MLP
# Number of GPUs: 512
# Packed Batch Size: 512
# Learning Rate: 2e-4
# Context Length: 16384
# Image Tile Threshold: 48
# ViT Drop Path: 0.0
# Weight Decay: 0.01
# Epoch: None
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_pretrain.py \
  --vision_path "/share/zwp/Model/InternVL/InternViT-300M-448px-V2_5" \
  --llm_path "/share/zwp/Model/Qwen/Qwen2.5-0.5B-Instruct" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/share/zwp/DATA/coco/internvl-pt/coco_caption.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --min_num_frame 8 \
  --max_num_frame 32 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --max_steps 100000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  --use_packed_ds False \
  --num_images_expected 48 \
  --max_packed_tokens 16384 \
  --max_buffer_size 20 \
  --log_freq 1000 \
  --strict_mode False \
  --replacement False \
  --allow_overflow False \
  --remove_unused_columns False \
  --loss_reduction "square" \
  --loss_reduction_all_gather True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
