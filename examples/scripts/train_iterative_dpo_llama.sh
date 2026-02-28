set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

PROJECT_NAME=llama-3.1-8b-iter-dpo
mkdir -p ./checkpoint/${PROJECT_NAME}
GENERATE_OUTPUT=./checkpoint/${PROJECT_NAME}/generate.jsonl
RM_OUTPUT=./checkpoint/${PROJECT_NAME}/rm.jsonl
MODEL_OUTPUT_PATH=./checkpoint/${PROJECT_NAME}/checkpoint
LOG_PATH=./checkpoint/${PROJECT_NAME}/tb_log
ITER_LOG_PATH=null

TRAINING_ITERS=3
ROLLOUT_BATCH_SIZE=20240

POLICY_MODEL_PATH=/seu_nvme/home/gengxin/220242297/hf_cache/models--allenai--Llama-3.1-Tulu-3-8B-SFT/snapshots/f2a0b46b0cfda21003c6141b1ff837b7e165524d
REF_MODEL_PATH=$POLICY_MODEL_PATH
RM_MODEL_PATH=OpenRLHF/Llama-3-8b-rm-mixture

iter=0
if [ -f $ITER_LOG_PATH ]; then
   iter=$(cat $ITER_LOG_PATH)
fi

while (($iter < $TRAINING_ITERS)); do
   echo "Iter: $iter"
   # Use latest model if past first iteration
   if ((iter > 0)); then
      POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
   fi

   GENERATE_OUTPUT=./checkpoint/${PROJECT_NAME}/generate_${iter}.jsonl
   RM_OUTPUT=./checkpoint/${PROJECT_NAME}/rm_${iter}.jsonl
   MODEL_OUTPUT_PATH=./checkpoint/${PROJECT_NAME}/checkpoint_${iter}

   read -r -d '' generate_commands <<EOF
openrlhf.cli.batch_inference
   --eval_task generate_vllm \
   --pretrain $POLICY_MODEL_PATH \
   --max_new_tokens 2048 \
   --prompt_max_len 2048 \
   --dataset HuggingFaceH4/ultrafeedback_binarized \
   --input_key prompt \
   --dataset_split train_prefs \
   --apply_chat_template \
   --temperature 1.0 \
   --tp_size 4 \
   --best_of_n 8 \
   --enable_prefix_caching \
   --max_num_seqs 512 \
   --iter $iter \
   --rollout_batch_size $ROLLOUT_BATCH_SIZE \
   --output_path $GENERATE_OUTPUT
EOF
   echo $generate_commands
   python -m $generate_commands
   checkSuccess "GENERATE"

   read -r -d '' get_rewards_commands <<EOF
openrlhf.cli.batch_inference
   --eval_task rm \
   --pretrain $RM_MODEL_PATH \
   --bf16 \
   --max_len 4096 \
   --dataset $GENERATE_OUTPUT  \
   --dataset_probs 1.0 \
   --zero_stage 0 \
   --post_processor iter_dpo \
   --micro_batch_size 4 \
   --output_path $RM_OUTPUT
EOF
   echo $get_rewards_commands
   deepspeed --module $get_rewards_commands
   checkSuccess "RM"

   read -r -d '' dpo_commands <<EOF
openrlhf.cli.train_dpo \
   --max_len 4096 \
   --dataset $RM_OUTPUT \
   --dataset_probs 1.0 \
   --prompt_key prompt \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain $POLICY_MODEL_PATH \
   --ref_pretrain $REF_MODEL_PATH \
   --save_path $MODEL_OUTPUT_PATH \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 5e-7 \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb 60b1d93e0cc89ca5895be545be917a31388c3b3c \
   --wandb_project iter_dpo \
   --wandb_group baseline \
   --wandb_run_name ${PROJECT_NAME}_ep${iter}
EOF
   echo $dpo_commands
   deepspeed --module $dpo_commands
   checkSuccess "DPO"

   echo "Starting evaluation..."
   CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
   SCRIPT_PATH="$CURRENT_DIR/../../run_eval.sh"
   bash "$SCRIPT_PATH" "$MODEL_OUTPUT_PATH" 1

   rm ${RM_OUTPUT}[0-9]*

   iter=$((iter + 1))
   if [[ "$ITER_LOG_PATH" != "null" ]]; then
      echo $iter >$ITER_LOG_PATH
   fi
done