set -x
export WANDB_MODE=offline

PROJECT_NAME=llama-3.1-8b-dpo
mkdir -p ./checkpoint/${PROJECT_NAME}
MODEL_OUTPUT_PATH=./checkpoint/${PROJECT_NAME}/checkpoint
LOG_PATH=./checkpoint/${PROJECT_NAME}/tb_log

POLICY_MODEL_PATH=allenai/Llama-3.1-Tulu-3-8B-SFT

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ${MODEL_OUTPUT_PATH} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 4 \
   --pretrain ${POLICY_MODEL_PATH} \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset HuggingFaceH4/ultrafeedback_binarized \
   --apply_chat_template \
   --train_split train_prefs \
   --eval_split test_prefs \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --packing_samples \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
