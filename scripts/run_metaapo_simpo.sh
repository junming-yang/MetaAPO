set -x
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline

checkSuccess() {
   if [[ $? != 0 ]]; then
      echo "FAILED $1"
      exit 1
   fi
}

BASE_SAVE_PATH=./checkpoint
PROJECT_NAME=metaapo-simpo-llama3-8b
mkdir -p ${BASE_SAVE_PATH}/checkpoint/${PROJECT_NAME}
PROJECT_PATH=${BASE_SAVE_PATH}/checkpoint/${PROJECT_NAME}
GENERATE_OUTPUT=${PROJECT_PATH}/generate.jsonl
RM_OUTPUT=${PROJECT_PATH}/rm.jsonl
MODEL_OUTPUT_PATH=${PROJECT_PATH}/checkpoint
LOG_PATH=${PROJECT_PATH}/tb_log

POLICY_MODEL_PATH=allenai/Llama-3.1-Tulu-3-8B-SFT
REF_POLICY_MODEL_PATH=$POLICY_MODEL_PATH
RM_MODEL_PATH=jmyang/llama3.1-8b-rm-ultrafeedback
DATASET_PATH=HuggingFaceH4/ultrafeedback_binarized
META_LEARNER_PATH=null

TRAINING_ITERS=3
ROLLOUT_BATCH_SIZE=20370
GPU_NUMS=4

iter=0

while (($iter < $TRAINING_ITERS)); do
    echo "Iter: $iter"
    if ((iter > 0)); then
        REF_POLICY_MODEL_PATH=$POLICY_MODEL_PATH
        POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
        prev_iter=$((iter - 1))
        if [ -d ${PROJECT_PATH}/checkpoint_${prev_iter}/meta_learner/step_latest ]; then
            META_LEARNER_PATH=${PROJECT_PATH}/checkpoint_${prev_iter}/meta_learner/step_latest
            echo "Load meta learner from ${META_LEARNER_PATH}"
        fi
    fi

    GENERATE_OUTPUT=${PROJECT_PATH}/generate_${iter}.jsonl
    RM_OUTPUT=${PROJECT_PATH}/rm_${iter}.jsonl
    MODEL_OUTPUT_PATH=${PROJECT_PATH}/checkpoint_${iter}
    IMPLICIT_REWARDS_OUTPUT=${PROJECT_PATH}/irm_${iter}.jsonl
    PROMPT_OUTPUT=${PROJECT_PATH}/prompt_${iter}.jsonl
    SAMPLED_DATASET=${PROJECT_PATH}/sampled_${iter}.jsonl
    MERGED_DATASET=${PROJECT_PATH}/merged_${iter}.jsonl

read -r -d '' cal_IRM_commands <<EOF
openrlhf.cli.cal_implicit_reward \
    --save_path ${MODEL_OUTPUT_PATH} \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 128 \
    --micro_train_batch_size 8 \
    --pretrain ${POLICY_MODEL_PATH} \
    --ref_pretrain ${REF_POLICY_MODEL_PATH} \
    --bf16 \
    --max_epochs 1 \
    --max_len 4096 \
    --zero_stage 3 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --dataset ${DATASET_PATH} \
    --apply_chat_template \
    --train_split train_prefs \
    --eval_split test_prefs \
    --chosen_key chosen \
    --rejected_key rejected \
    --flash_attn \
    --load_checkpoint \
    --gradient_checkpointing \
    --iter $iter \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --reward_output_file $IMPLICIT_REWARDS_OUTPUT \
    --loss_type simpo
EOF
    echo $cal_IRM_commands
    deepspeed --module $cal_IRM_commands
    checkSuccess "Cal Implicit Reward"

    # judge sampling
    python openrlhf/cli/sample.py \
        --meta_learner_path $META_LEARNER_PATH/../meta_learner.pt \
        --data_path $IMPLICIT_REWARDS_OUTPUT \
        --sample_output_path $SAMPLED_DATASET \
        --prompt_output_path $PROMPT_OUTPUT \
        --loss_type simpo

read -r -d '' generate_commands <<EOF
openrlhf.cli.batch_inference
    --eval_task generate_vllm_dp \
    --pretrain $POLICY_MODEL_PATH \
    --max_new_tokens 2048 \
    --prompt_max_len 2048 \
    --dataset $PROMPT_OUTPUT \
    --input_key prompt \
    --apply_chat_template \
    --temperature 1.0 \
    --tp_size $GPU_NUMS \
    --best_of_n 8 \
    --enable_prefix_caching \
    --max_num_seqs 512 \
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
    --output_path $RM_OUTPUT \
    --flash_attn
EOF
    echo $get_rewards_commands
    deepspeed --module $get_rewards_commands
    checkSuccess "RM"

    # merge dataset
    python -m openrlhf.cli.utils \
        --offline_path $SAMPLED_DATASET \
        --online_path $RM_OUTPUT \
        --merged_path $MERGED_DATASET

# training
read -r -d '' training_commands <<EOF
openrlhf.cli.train_meta_dpo \
    --save_path ${MODEL_OUTPUT_PATH} \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 128 \
    --micro_train_batch_size 2 \
    --pretrain ${POLICY_MODEL_PATH} \
    --ref_pretrain ${REF_POLICY_MODEL_PATH} \
    --bf16 \
    --max_epochs 1 \
    --max_len 4096 \
    --zero_stage 3 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --dataset ${MERGED_DATASET} \
    --prompt_key prompt \
    --chosen_key chosen \
    --rejected_key rejected \
    --online_chosen_key online_chosen \
    --online_rejected_key online_rejected \
    --flash_attn \
    --packing_samples \
    --load_checkpoint \
    --adam_offload \
    --gradient_checkpointing \
    --ref_offload \
    --meta_learner_ckpt ${META_LEARNER_PATH} \
    --loss_type simpo \
    --meta_k 8
EOF
    echo $training_commands
    deepspeed --module $training_commands
    checkSuccess "Meta DPO"

    rm ${RM_OUTPUT}[0-9]*

    iter=$((iter + 1))
    if [[ "$ITER_LOG_PATH" != "null" ]]; then
        echo $iter >$ITER_LOG_PATH
    fi
done
