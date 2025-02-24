#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from numpy import *
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer,LlamaForCausalLM,AutoTokenizer
# from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
import utils
import numpy as np
from transformers import AutoModelForCausalLM, LlamaConfig
import os
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT_DICT_tulu = {
    "prompt_input": (
        "### Input:\n{input}\n\n### Response:"
    )
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    mode: Optional[str] = field(default="base")
    grouping_idx: Optional[int] = field(default=4)
    grouping_begin_idx: Optional[int] = field(default=16)
    grouping_end_idx: Optional[int] = field(default=33)
    lora_hidden: Optional[int] = field(default=4)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_layer_idx: int = field(default=0)
    use_group_idx: int = field(default=None)
    save_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = np.copy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=input_ids)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if "infinity" in data_path:
            list_data_dict = utils.jload(data_path)
            prompt_input = PROMPT_DICT_tulu["prompt_input"]
        elif "pmc_" in data_path:
            list_data_dict = utils.jload(data_path)
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # list_data_dict = list_data_dict[:100]
        print(f"Loaded {len(list_data_dict)} examples.")

        logging.warning("Formatting inputs...")
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

    if model_args.mode == "base":
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        from model_file.modeling_llama_weight import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map='auto',
            config=config,
        )
        print("load modeling_llama_weight LlamaForCausalLM")
    elif model_args.mode == "softmax_init_sequential":
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        config.grouping_idx = model_args.grouping_idx
        config.grouping_begin_idx = model_args.grouping_begin_idx
        config.grouping_end_idx = model_args.grouping_end_idx
        from model_file.modeling_llama_softmax_layer_init_weight_by_index import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map='auto',
            config=config,
        )
        print("load modeling_llama_softmax_layer_init_weight_by_index LlamaForCausalLM")
        if training_args.use_layer_idx > 0:
            init_weight_list = []
            for i in range(1, training_args.use_layer_idx+1):
                init_weight_list.append(np.load(f"{training_args.save_dir}/init_weights_sequential_index_{i}.npy"))
            init_weight_list = np.array(init_weight_list)
            model.init_layer_by_idx(training_args.use_layer_idx)
            model.init_v2_proj_by_index(init_weight_list, training_args.use_layer_idx)
        else:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    result_o1 = torch.zeros(32, 1024, 4096).to(model.device)
    all_hidden_states_v2 = torch.zeros(32, 1024, 4096).to(model.device)
    result_o2 = torch.zeros(32, 1024, 4096).to(model.device)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    for i in range(len(data_module["train_dataset"].input_ids)):
        print(i)
        input_token = data_module["train_dataset"].input_ids[i].unsqueeze(0)
        input_token = input_token.to(model.device)
        with torch.no_grad(): 
            outpus = model(input_ids=input_token)
        if model_args.mode == "base":
            hidden_states = torch.cat(outpus.hidden_states, dim=0)
            result_o1 = result_o1 + hidden_states
        else:
            attentions = torch.cat(outpus.attentions, dim=0)
            hidden_states_v2 = torch.cat(outpus.hidden_states, dim=0)
            result_o2 = result_o2 + attentions
            all_hidden_states_v2 = all_hidden_states_v2 + hidden_states_v2
    if model_args.mode == "base":
        result_o1 = result_o1.cpu().detach().numpy()
        result_o1 = result_o1/len(data_module["train_dataset"].input_ids)
        np.save('{}/result_base_output.npy'.format(training_args.save_dir), result_o1)
    elif model_args.mode == "softmax_init_sequential":
        result_o2 = result_o2.cpu().detach().numpy()
        result_o2 = result_o2/len(data_module["train_dataset"].input_ids)
        all_hidden_states_v2 = all_hidden_states_v2.cpu().detach().numpy()
        all_hidden_states_v2 = all_hidden_states_v2/len(data_module["train_dataset"].input_ids)
        np.save('{}/result_sequential_index_{}.npy'.format(training_args.save_dir,training_args.use_layer_idx), result_o2)
        np.save('{}/all_hidden_states_v2_sequential_index_{}.npy'.format(training_args.save_dir,training_args.use_layer_idx), all_hidden_states_v2)



if __name__ == "__main__":
    train()
