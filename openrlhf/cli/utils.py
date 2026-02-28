import json
import argparse
from collections import defaultdict

import torch
import warnings


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys if present"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def load_checkpoint(path):
    """Load checkpoint with proper error handling"""
    try:
        # Ignore warning messages
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(path, map_location='cpu')
            
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            else:
                return checkpoint
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def refine_prompt(evolve_prompt_path, refine_prompt_path):
    prompts = []
    with open(evolve_prompt_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            # Check if there is a termination symbol
            if "<|im_end|>" in data["output"]:
                prompt = data["output"].split("<|im_end|>")[0].strip()
            elif "<|end_of_text|>" in data["output"]:
                prompt = data["output"].split("<|end_of_text|>")[0].strip()
            else:
                prompt = data["output"].strip()
            prompts.append(prompt)

    with open(refine_prompt_path, "w") as f:
        for prompt in prompts:
            json.dump({"prompt": prompt}, f, ensure_ascii=False)
            f.write("\n")


def merge_dataset(offline_path, online_path, merged_path, use_qwen_format=False, unsampled_path=None):
    merged_data = {}
    stats = defaultdict(int)
    
    with open(offline_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = data.get('prompt', '')
                if not prompt:
                    continue
                if use_qwen_format:
                    prompt_fmt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    chosen = data.get('chosen', '') + "<|im_end|>"
                    rejected = data.get('rejected', '') + "<|im_end|>"
                else:
                    prompt_fmt = f"<|user|>\n{prompt}\n<|assistant|>\n"
                    chosen = data.get('chosen', '') + "<|end_of_text|>"
                    rejected = data.get('rejected', '') + "<|end_of_text|>"
                if prompt_fmt not in merged_data:
                    merged_data[prompt_fmt] = {
                        'prompt': prompt_fmt,
                        'chosen': chosen,
                        'rejected': rejected,
                        'online_chosen': '',
                        'online_rejected': ''
                    }
                stats['un_add_token_count'] += 1
            except json.JSONDecodeError:
                stats['un_add_token_errors'] += 1
                continue

    with open(online_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = data.get('prompt', '')
                if not prompt:
                    continue
                if prompt in merged_data:
                    merged_data[prompt]['online_chosen'] = data.get('chosen', '')
                    merged_data[prompt]['online_rejected'] = data.get('rejected', '')
                    stats['matched_count'] += 1
                stats['un_generate_count'] += 1
            except json.JSONDecodeError:
                stats['un_generate_errors'] += 1
                continue
        
    # add unsampled data as offline data, use chosen and rejected as chosen and online chosen
    if unsampled_path is not None:
        with open(unsampled_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    prompt = data.get('prompt', '')
                    if not prompt:
                        continue
                    if prompt not in merged_data:
                        merged_data[prompt] = {
                            'prompt': prompt,
                            'chosen': data.get('chosen', ''),
                            'rejected': data.get('rejected', ''),
                            'online_chosen': data.get('chosen', ''),
                            'online_rejected': data.get('rejected', '')
                        }
                        stats['unsampled_add_count'] += 1
                except json.JSONDecodeError:
                    stats['unsampled_error'] += 1
                    continue

    with open(merged_path, 'w', encoding='utf-8') as f:
        for data in merged_data.values():
            if data['chosen'] and data['rejected'] and data['online_chosen'] and data['online_rejected']:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                stats['valid_merged_count'] += 1
        

def merge_dataset_adpo(offline_path, online_path, merged_path, use_qwen_format=False):
    stats = defaultdict(int)
    with open(merged_path, 'w', encoding='utf-8') as fout:
        # Process offline data
        with open(offline_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    prompt = data.get('prompt', '')
                    if not prompt:
                        continue
                    if use_qwen_format:
                        prompt_fmt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                        chosen = data.get('chosen', '') + "<|im_end|>"
                        rejected = data.get('rejected', '') + "<|im_end|>"
                    else:
                        prompt_fmt = f"<|user|>\n{prompt}\n<|assistant|>\n"
                        chosen = data.get('chosen', '') + "<|end_of_text|>"
                        rejected = data.get('rejected', '') + "<|end_of_text|>"
                    out = {
                        'prompt': prompt_fmt,
                        'chosen': chosen,
                        'rejected': rejected
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                    stats['offline_count'] += 1
                except json.JSONDecodeError:
                    stats['offline_error'] += 1
                    continue

        # Process online data
        with open(online_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    prompt = data.get('prompt', '')
                    chosen = data.get('chosen', '')
                    rejected = data.get('rejected', '')
                    if not prompt:
                        continue
                    out = {
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                    stats['online_count'] += 1
                except json.JSONDecodeError:
                    stats['online_error'] += 1
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolve_prompt_path", type=str, help="Path to the evolve prompt file")
    parser.add_argument("--refine_prompt_path", type=str, help="Path to the refine prompt file")
    parser.add_argument("--offline_path", type=str, help="Path to the offline dataset")
    parser.add_argument("--online_path", type=str, help="Path to the online dataset")
    parser.add_argument("--merged_path", type=str, help="Path to save the merged dataset")
    parser.add_argument("--adpo", action="store_true", help="Whether to use ADPO")
    parser.add_argument("--use_qwen_format", action="store_true", help="Whether to use Qwen chat format")
    parser.add_argument("--unsampled_path", type=str, help="Path to the unsampled dataset", default=None)
    
    args = parser.parse_args()
    
    if all([args.offline_path, args.online_path, args.merged_path]):
        if args.adpo:
            merge_dataset_adpo(args.offline_path, args.online_path, args.merged_path, use_qwen_format=args.use_qwen_format)
        else:
            merge_dataset(args.offline_path, args.online_path, args.merged_path, use_qwen_format=args.use_qwen_format, unsampled_path=args.unsampled_path)
    elif all([args.evolve_prompt_path, args.refine_prompt_path]):
        refine_prompt(args.evolve_prompt_path, args.refine_prompt_path)
    else:
        parser.error("Either provide all merge dataset arguments or all refine prompt arguments")