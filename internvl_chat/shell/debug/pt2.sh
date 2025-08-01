set -x

export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA="mlx5_1,mlx5_2,mlx5_3,mlx5_4"
# export NCCL_IB_HCA="mlx5"
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
export PDSH_SSH_ARGS_APPEND="-p52022"
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
accelerate launch \
  --config_file "shell/debug/config-1.yaml" \
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
  --drop_path_rate 0.1 \
  --min_num_frame 8 \
  --max_num_frame 32 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --max_steps 22000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  --use_packed_ds True \
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
