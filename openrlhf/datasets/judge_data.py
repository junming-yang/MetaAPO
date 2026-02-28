import datasets
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

def load_dataset(path: str, split: str = None) -> datasets.Dataset:
    """加载数据集"""
    if path.endswith(".jsonl") or path.endswith(".json"):
        dataset = datasets.load_dataset("json", data_files=path, split=split)
    else:
        dataset = datasets.load_dataset(path, split=split)
    return dataset

def process_conversation(text: str) -> Tuple[str, str]:
    """处理对话文本，分离出 prompt 和 response"""
    if "<|user|>\n" in text:
        text = text.split("<|user|>\n")[-1]
    elif "user\n" in text:
        text = text.split("user\n")[-1]
    
    if "<|assistant|>\n" in text:
        parts = text.split("<|assistant|>\n")
    else:
        parts = text.split("assistant\n")
    
    if len(parts) != 2:
        return None, None
    
    prompt = parts[0].strip()
    response = parts[1].strip()
    
    return prompt, response

def process_reward_data(reward_path: str, output_path: str):
    """处理奖励数据，提取所需信息并保存"""
    print("Loading reward dataset...")
    reward_dataset = load_dataset(reward_path, "train")

    # 一次性加载所有数据到内存
    reward_data = {
        "chosen_text": reward_dataset["chosen_text"],
        "rejected_text": reward_dataset["rejected_text"],
        "chosen_logp": reward_dataset["chosen_logp"],
        "rejected_logp": reward_dataset["rejected_logp"],
        "ref_chosen_logp": reward_dataset["ref_chosen_logp"],
        "ref_rejected_logp": reward_dataset["ref_rejected_logp"]
    }
    
    processed_data = []
    print("Processing reward dataset...")
    for i in tqdm(range(len(reward_dataset)), desc="Processing conversations"):
        chosen_prompt, chosen_response = process_conversation(reward_data["chosen_text"][i])
        rejected_prompt, rejected_response = process_conversation(reward_data["rejected_text"][i])
        
        if None in [chosen_prompt, chosen_response, rejected_prompt, rejected_response]:
            continue
            
        # 确保 prompt 相同
        if chosen_prompt != rejected_prompt:
            continue
            
        chosen_reward = reward_data["chosen_logp"][i] - reward_data["ref_chosen_logp"][i]
        rejected_reward = reward_data["rejected_logp"][i] - reward_data["ref_rejected_logp"][i]
        reward = chosen_reward - rejected_reward
        
        processed_data.append({
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "chosen_logp": reward_data["chosen_logp"][i],
            "rejected_logp": reward_data["rejected_logp"][i],
            "ref_chosen_logp": reward_data["ref_chosen_logp"][i],
            "ref_rejected_logp": reward_data["ref_rejected_logp"][i],
            "reward": reward
        })
    
    # 保存为 jsonl 文件
    print(f"Saving {len(processed_data)} samples to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
def mix_dataset(confident_data_path, unconfident_data_path, generated_data_path, mix_ratio=0.5):
    """混合数据集"""
    confident_dataset = load_dataset(confident_data_path, "train")
    unconfident_dataset = load_dataset(unconfident_data_path, "train")
    generated_dataset = load_dataset(generated_data_path, "train")
    
def merge_jsonl_files(file1, file2, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Process first file
        with open(file1, 'r', encoding='utf-8') as f1:
            for line in f1:
                # Verify valid JSON
                try:
                    json.loads(line)
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file1}")
                    
        # Process second file
        with open(file2, 'r', encoding='utf-8') as f2:
            for line in f2:
                # Verify valid JSON
                try:
                    json.loads(line)
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file2}")
                    
def build_chat_format(dataset_path, output_path):
    dataset = load_dataset(dataset_path, "train")
    dataset_dict = dataset.to_dict()
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(dataset_dict['prompt'])):
            sample = {
                'prompt': [{'content': dataset_dict['prompt'][i], 'role': 'user'}],
                'chosen': [{'content': dataset_dict['prompt'][i], 'role': 'user'}, {'content': dataset_dict['chosen'][i], 'role': 'assistant'}],
                'rejected': [{'content': dataset_dict['prompt'][i], 'role': 'user'}, {'content': dataset_dict['rejected'][i], 'role': 'assistant'}],
                'chosen_logp': dataset_dict['chosen_logp'][i],
                'rejected_logp': dataset_dict['rejected_logp'][i],
                'ref_chosen_logp': dataset_dict['ref_chosen_logp'][i],
                'ref_rejected_logp': dataset_dict['ref_rejected_logp'][i]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

# 根据 reward 的值，提取满足条件的部分，保存到数据集中
def split_reward_dataset(dataset_path, output_path, greater_threshold=None, less_threshold=None):
    loaded_dataset = load_dataset(dataset_path, "train")
    if greater_threshold is not None:
        greater_dataset = [item for item in loaded_dataset if item['reward'] > greater_threshold]
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in greater_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    if less_threshold is not None:
        less_dataset = [item for item in loaded_dataset if item['reward'] < less_threshold]
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in less_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
def add_chat_tokens(dataset_path, output_path):
    dataset = load_dataset(dataset_path, "train")
    
    # 将数据集转换为列表并添加标记
    processed_data = []
    for item in dataset:
        processed_item = item.copy()
        processed_item['prompt'] = '<|user|>\n' + item['prompt'] + '\n<|assistant|>\n'
        processed_item['chosen'] = item['chosen'] + '<|end_of_text|>'
        processed_item['rejected'] = item['rejected'] + '<|end_of_text|>'
        processed_data.append(processed_item)
    
    # 保存到新的数据集中
    print(f"Saving {len(processed_data)} samples to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 设置路径
    # implicit_reward_path = "./data/implicit_rewards.jsonl"
    # output_path = "./data/processed_judge_data.jsonl"
    
    # # 处理数据并保存
    # process_reward_data(implicit_reward_path, output_path)
    
    # # 加载数据
    # dataset = load_dataset(output_path, "train")
    # # 将reward 小于4的样本保存到./data/low_reward_samples.jsonl，大于4的样本保存到./data/high_reward_samples.jsonl
    # low_reward_samples = [item for item in dataset if item['reward'] < 4]
    # high_reward_samples = [item for item in dataset if item['reward'] >= 4]
    # with open('./data/low_reward_samples.jsonl', 'w', encoding='utf-8') as f:
    #     for item in low_reward_samples:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # with open('./data/high_reward_samples.jsonl', 'w', encoding='utf-8') as f:
    #     for item in high_reward_samples:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # build_chat_format("./data/learned_samples.jsonl", "./data/learned_samples_chat.jsonl")
    add_chat_tokens("./data/low_reward_samples.jsonl", "./data/low_reward_samples_chat_tokens.jsonl")