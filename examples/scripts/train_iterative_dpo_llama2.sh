set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

PROJECT_NAME=llama-3-8b-iter-dpo-rm-31-re
mkdir -p ./checkpoint/${PROJECT_NAME}
# GENERATE_OUTPUT=./checkpoint/${PROJECT_NAME}/generate.jsonl
# RM_OUTPUT=./checkpoint/${PROJECT_NAME}/rm.jsonl
# MODEL_OUTPUT_PATH=./checkpoint/${PROJECT_NAME}/checkpoint
LOG_PATH=./checkpoint/${PROJECT_NAME}/tb_log
ITER_LOG_PATH=null

TRAINING_ITERS=3
ROLLOUT_BATCH_SIZE=20240

POLICY_MODEL_PATH=/seu_nvme/home/gengxin/220242297/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
REF_MODEL_PATH=$POLICY_MODEL_PATH

# RM_MODEL_PATH=opencompass/CompassJudger-1-1.5B-Instruct
RM_MODEL_PATH=OpenRLHF/Llama-3-8b-rm-mixture

iter=2
if [ -f $ITER_LOG_PATH ]; then
   iter=$(cat $ITER_LOG_PATH)
fi

while (($iter < $TRAINING_ITERS)); do
   echo "Iter: $iter"
   # Use latest model if past first iteration
   if ((iter > 0)); then
      # POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
      POLICY_MODEL_PATH=./checkpoint/${PROJECT_NAME}/checkpoint_1
   fi

   GENERATE_OUTPUT=./checkpoint/${PROJECT_NAME}/generate_iter${iter}.jsonl
   RM_OUTPUT=./checkpoint/${PROJECT_NAME}/rm_iter${iter}.jsonl
   MODEL_OUTPUT_PATH=./checkpoint/${PROJECT_NAME}/checkpoint_${iter}

   if [ "$iter" -ne 0 ]; then
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
   --tp_size 8 \
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
   fi

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
   --gradient_checkpointing \
   --use_tensorboard $LOG_PATH
EOF
   echo $dpo_commands
   deepspeed --module $dpo_commands
   checkSuccess "DPO"

   iter=$((iter + 1))
   if [[ "$ITER_LOG_PATH" != "null" ]]; then
      echo $iter >$ITER_LOG_PATH
   fi
done