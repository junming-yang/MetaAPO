from openrlhf.models.meta_learner import MetaLearner
import argparse
import datasets
import random
import json
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

from openrlhf.cli.utils import remove_module_prefix, load_checkpoint


random.seed(42)

def load_dataset(path: str, split: str = None):
    """Load dataset"""
    if path.endswith(".jsonl") or path.endswith(".json"):
        dataset = datasets.load_dataset("json", data_files=path, split=split)
    else:
        dataset = datasets.load_dataset(path, split=split)
    return dataset

# delete chat template
def remove_chat_template(example):
    chosen_text = example['chosen_text']
    rejected_text = example['rejected_text']
    
    if "\nuser\n" in chosen_text:
        prompt = chosen_text.split("\nuser\n")[1].split("\nassistant\n")[0].strip()
        if len(chosen_text.split("\nassistant\n")) > 1:
            chosen = chosen_text.split("\nassistant\n")[1].strip()
        else:
            chosen = ""
        if len(rejected_text.split("\nassistant\n")) > 1:
            rejected = rejected_text.split("\nassistant\n")[1].strip()
        else:
            rejected = ""
    elif "<|user|>\n" in chosen_text:
        prompt = chosen_text.split("<|user|>\n")[1].split("\n<|assistant|>\n")[0].strip()
        if len(chosen_text.split("\n<|assistant|>\n")) > 1:
            chosen = chosen_text.split("\n<|assistant|>\n")[1].strip()
        else:
            chosen = ""
        if len(rejected_text.split("\n<|assistant|>\n")) > 1:
            rejected = rejected_text.split("\n<|assistant|>\n")[1].strip()
        else:
            rejected = ""
    elif "<|im_start|>user\n" in chosen_text:
        prompt = chosen_text.split("<|im_start|>user\n")[1].split("<|im_end|>\n<|im_start|>assistant\n")[0].strip()
        if len(chosen_text.split("<|im_end|>\n<|im_start|>assistant\n")) > 1:
            chosen = chosen_text.split("<|im_end|>\n<|im_start|>assistant\n")[1].strip()
        else:
            chosen = ""
        if len(rejected_text.split("<|im_end|>\n<|im_start|>assistant\n")) > 1:
            rejected = rejected_text.split("<|im_end|>\n<|im_start|>assistant\n")[1].strip()
        else:
            rejected = ""
    else:
        prompt = chosen_text
    
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected, **{k: v for k, v in example.items() if k not in ["chosen_text", "rejected_text"]}}


def sample_dataset(example, meta_learner):
    loss_data = F.logsigmoid(torch.tensor(example['reward']))
    prob = float(meta_learner.predict(loss_data))
    if random.random() > prob:
        res = {
            'prompt': example['prompt'],
            'chosen': example['chosen'],
            'rejected': example['rejected'],
            'chosen_logp': example['chosen_logp'],
            'ref_chosen_logp': example['ref_chosen_logp'],
            'rejected_logp': example['rejected_logp'],
            'ref_rejected_logp': example['ref_rejected_logp'],
            'reward': example['reward'],
            'prob': prob,
        }
        return res
    else:
        return None


def adaptive_sample_dataset(args):
    max_count_coeff=0.8
    dataset = args.data_path
    save_dir = args.prompt_output_path
    samples_dir = args.sample_output_path
    unsampled_dir = args.unsampled_output_path
    
    split = 'train'
    
    # load meta learner
    meta_learner = MetaLearner()
    if args.meta_learner_path and 'null' not in args.meta_learner_path:
        state_dict = load_checkpoint(args.meta_learner_path)
        state_dict = remove_module_prefix(state_dict)
        meta_learner.load_state_dict(state_dict)
    
    data = load_dataset(dataset, split=split)
    
    train_data = data.map(remove_chat_template)
    
    max_count = len(train_data) * max_count_coeff
    
    sampled_data = []
    prompt_data = []
    unsampled_data = []
    # calculate the probability of each sample
    for i in tqdm(range(len(train_data))):
        loss_data = F.logsigmoid(torch.tensor(train_data[i]['reward']))
        prob = float(meta_learner.predict(loss_data))
        # random sample with probability
        if args.random_sample:
            prob = random.random()
        
        if random.random() > prob or args.all_sample:
            if args.all_sample:
                flag = True if random.random() > prob else False
                prompt_data.append({'prompt': train_data[i]['prompt'], 'prob': prob, 'label': flag})
            else:
                prompt_data.append({'prompt': train_data[i]['prompt'], 'prob': prob})
            if args.loss_type == "dpo":
                sampled_data.append({
                    'prompt': train_data[i]['prompt'], 
                    'chosen': train_data[i]['chosen'], 
                    'rejected': train_data[i]['rejected'],
                    'chosen_logp': train_data[i]['chosen_logp'],
                    'ref_chosen_logp': train_data[i]['ref_chosen_logp'],
                    'rejected_logp': train_data[i]['rejected_logp'],
                    'ref_rejected_logp': train_data[i]['ref_rejected_logp'],
                    'reward': train_data[i]['reward'],
                })
            elif args.loss_type == "simpo":
                sampled_data.append({
                    'prompt': train_data[i]['prompt'], 
                    'chosen': train_data[i]['chosen'], 
                    'rejected': train_data[i]['rejected'],
                    'chosen_logp': train_data[i]['chosen_logp'],
                    'rejected_logp': train_data[i]['rejected_logp'],
                    'reward': train_data[i]['reward'],
                })
        else:
            if args.loss_type == "dpo":
                unsampled_data.append({
                    'prompt': train_data[i]['prompt'], 
                    'chosen': train_data[i]['chosen'], 
                    'rejected': train_data[i]['rejected'],
                    'chosen_logp': train_data[i]['chosen_logp'],
                    'ref_chosen_logp': train_data[i]['ref_chosen_logp'],
                    'rejected_logp': train_data[i]['rejected_logp'],
                    'ref_rejected_logp': train_data[i]['ref_rejected_logp'],
                    'reward': train_data[i]['reward'],
                })
            elif args.loss_type == "simpo":
                unsampled_data.append({
                    'prompt': train_data[i]['prompt'], 
                    'chosen': train_data[i]['chosen'], 
                    'rejected': train_data[i]['rejected'],
                    'chosen_logp': train_data[i]['chosen_logp'],
                    'rejected_logp': train_data[i]['rejected_logp'],
                    'reward': train_data[i]['reward'],
                })

    with open(save_dir, 'w', encoding='utf-8') as f:
        for data in prompt_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    with open(samples_dir, 'w', encoding='utf-8') as f:
        for data in sampled_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"sample ratio: {100 * (len(prompt_data) / len(train_data)):.2f}%")

    if unsampled_dir is not None:
        with open(unsampled_dir, 'w', encoding='utf-8') as f:
            for data in unsampled_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_learner_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample_output_path", type=str, required=True)
    parser.add_argument("--prompt_output_path", type=str, required=True)
    parser.add_argument("--unsampled_output_path", type=str, default=None)
    parser.add_argument("--max_count_coeff", type=float, default=0.8)
    parser.add_argument("--loss_type", type=str, default="dpo")
    parser.add_argument("--random_sample", action="store_true", default=False)
    parser.add_argument("--all_sample", action="store_true", default=False)
    args = parser.parse_args()
    
    adaptive_sample_dataset(args)