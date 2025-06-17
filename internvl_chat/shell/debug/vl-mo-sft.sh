set -x

export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA="mlx5_1,mlx5_2,mlx5_3,mlx5_4"
# export NCCL_IB_HCA="mlx5"
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
# export PDSH_SSH_ARGS_APPEND="-p52022"
# export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

TRAIN_VER=$1

OUTPUT_DIR="/share/zwp/WORK_DIR/internvl_debug/vl-mo_sft_${TRAIN_VER}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

accelerate launch \
  --config_file "shell/debug/config-1.yaml" \
  internvl/train/internvl_motion_finetune.py \
  --model_name_or_path "/share/pretrained/mllm/InternVL/InternVL3-1B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "/share/lh/code/vl-motion-dev/shell/debug/egodex_sft_meta.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --motion_unit_length 32 \
  --nb_code 1024 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
