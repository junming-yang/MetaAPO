from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.datasets.prompt import evolve_strategies

def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, evolve_strategy=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            if evolve_strategy:
                chat = evolve_strategy(chat)
            chat = [{"role": "user", "content": chat}]
        else:
            if evolve_strategy:
                assert isinstance(chat, list)
                chat = [{"role": "user", "content": evolve_strategy(chat[0]["content"])}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        evolve_strategy=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        
        if evolve_strategy:
            # 1 is for breadth, 2 is for deepen, 3 is for constraints, 4 is for concretizing, 5 is for reasoning
            self.evolve_strategy = evolve_strategies[evolve_strategy - 1]
        else:
            self.evolve_strategy = None

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template, evolve_strategy=self.evolve_strategy)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
