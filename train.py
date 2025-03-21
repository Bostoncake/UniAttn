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

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer,AutoModelForCausalLM,LlamaConfig,MistralConfig,Gemma2Config
import numpy as np
from utils import data_utils
import os
from transformers import AutoModelForCausalLM, LlamaConfig

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
    grouping_idx: Optional[int] = field(default=4)
    grouping_begin_idx: Optional[int] = field(default=16)
    grouping_end_idx: Optional[int] = field(default=33)
    model_mode: Optional[str] = field(default="plain")
    proj_init_weight_path: Optional[str] = field(default="init_weight_path")


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
    train_mode: Optional[str] = field(default="one_stage")


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
            padding="longest",
            max_length=tokenizer.model_max_length,
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
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if "infinity" in data_path:
            print("load dataset infinity")
            list_data_dict = data_utils.jload(data_path)
            prompt_input = PROMPT_DICT_tulu["prompt_input"]
        else:
            try:
                list_data_dict = data_utils.load_jsonl(data_path)
            except:
                list_data_dict = data_utils.jload(data_path)
            print("load dataset pcm")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
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
    if "Llama" in model_args.model_name_or_path:
        if model_args.model_mode=="plain":
            from model_file.modeling_llama import LlamaForCausalLM
            print("load from model_file.modeling_llama import LlamaForCausalLM")

            config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=config,
            )
        elif model_args.model_mode=="softmax":
            from model_file.modeling_llama_softmax import LlamaForCausalLM
            print("load from model_file.modeling_llama_softmax import LlamaForCausalLM")
            config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
            config.grouping_idx = model_args.grouping_idx
            config.grouping_begin_idx = model_args.grouping_begin_idx
            config.grouping_end_idx = model_args.grouping_end_idx
            print(config)
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=config,
            )
        elif model_args.model_mode=="UniAttn":
            from model_file.modeling_llama_softmax_layer_init import LlamaForCausalLM
            print("load from model_file.modeling_llama_softmax_layer_init import LlamaForCausalLM")
            config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
            config.grouping_idx = model_args.grouping_idx
            config.grouping_begin_idx = model_args.grouping_begin_idx
            config.grouping_end_idx = model_args.grouping_end_idx
            model = LlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    config=config,
                )
            print("init weight")
            init_weight_list = np.load(model_args.proj_init_weight_path)
            model.init_v2_proj(init_weight_list)

    elif "Mistral" in model_args.model_name_or_path:
        if model_args.model_mode=="plain":
            from model_file.modeling_mistral import MistralForCausalLM
            print("load from model_file.modeling_mistral import MistralForCausalLM")

            config = MistralConfig.from_pretrained(model_args.model_name_or_path)
            model = MistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=config,
            )
        elif model_args.model_mode=="softmax":
            from model_file.modeling_mistral_softmax import MistralForCausalLM
            print("load from model_file.modeling_llama_softmax import MistralForCausalLM")
            config = MistralConfig.from_pretrained(model_args.model_name_or_path)
            config.grouping_idx = model_args.grouping_idx
            config.grouping_begin_idx = model_args.grouping_begin_idx
            config.grouping_end_idx = model_args.grouping_end_idx
            print(config)
            model = MistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=config,
            )
        elif model_args.model_mode=="UniAttn":
            from model_file.modeling_llama_softmax_layer_init import MistralForCausalLM
            print("load from model_file.modeling_llama_softmax_layer_init import MistralForCausalLM")
            config = MistralConfig.from_pretrained(model_args.model_name_or_path)
            config.grouping_idx = model_args.grouping_idx
            config.grouping_begin_idx = model_args.grouping_begin_idx
            config.grouping_end_idx = model_args.grouping_end_idx
            model = MistralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    config=config,
                )
            print("init weight")
            init_weight_list = np.load(model_args.proj_init_weight_path)
            model.init_v2_proj(init_weight_list)

    elif "gemma" in model_args.model_name_or_path or "Gemma" in model_args.model_name_or_path:
        if model_args.model_mode=="plain":
            from model_file.modeling_gemma2 import Gemma2ForCausalLM
            print("load from model_file.modeling_gemma2 import Gemma2ForCausalLM")
            config = Gemma2Config.from_pretrained(model_args.model_name_or_path)
            model = Gemma2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=config,
            )
        elif model_args.model_mode=="softmax":
            from model_file.modeling_gemma2_softmax import Gemma2ForCausalLM
            print("load from model_file.modeling_gemma2_softmax import Gemma2ForCausalLM")
            config = Gemma2Config.from_pretrained(model_args.model_name_or_path)
            config.grouping_idx = model_args.grouping_idx
            config.grouping_begin_idx = model_args.grouping_begin_idx
            config.grouping_end_idx = model_args.grouping_end_idx
            print(config)
            model = Gemma2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                config=config,
            )
        elif model_args.model_mode=="UniAttn":
            from model_file.modeling_gemma2_softmax_layer_init import Gemma2ForCausalLM
            print("load from model_file.modeling_gemma2_softmax_layer_init import Gemma2ForCausalLM")
            config = Gemma2Config.from_pretrained(model_args.model_name_or_path)
            config.grouping_idx = model_args.grouping_idx
            config.grouping_begin_idx = model_args.grouping_begin_idx
            config.grouping_end_idx = model_args.grouping_end_idx
            model = Gemma2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    config=config,
                )
            print("init weight")
            init_weight_list = np.load(model_args.proj_init_weight_path)
            model.init_v2_proj(init_weight_list)




    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
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
    print(model)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if training_args.train_mode == "one_stage":
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        try:
            trainer.train(resume_from_checkpoint=True)
        except:
            trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
    else:

        ori_output_dir = training_args.output_dir
        training_args.output_dir = os.path.join(ori_output_dir, "stage1_linear_ft")
        enable_tuning_name = []
        for name, param in model.model.named_parameters():
            if "v2_proj" in name:
                enable_tuning_name.append(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Modules enabled tuning in the 1st-stage: ", enable_tuning_name)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        try:
            trainer.train(resume_from_checkpoint=True)
        except:
            trainer.train()
        trainer.save_model(output_dir=training_args.output_dir)
        first_stage_dir=training_args.output_dir

        stage2_full_ft_name = "stage2_full_ft"
        training_args.output_dir = os.path.join(ori_output_dir, stage2_full_ft_name)
        config = LlamaConfig.from_pretrained(first_stage_dir)
        config.grouping_idx = model_args.grouping_idx
        config.grouping_begin_idx = model_args.grouping_begin_idx
        config.grouping_end_idx = model_args.grouping_end_idx
        print(config)
        from model_file.modeling_llama_softmax_layer_init import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            first_stage_dir,
            cache_dir=training_args.cache_dir,
            config=config,
        )
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
        # enable full fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        print("Enable full fine-tuning in the 2nd-stage.")
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        try:
            trainer.train(resume_from_checkpoint=True)
        except:
            trainer.train()
        trainer.save_model(output_dir=training_args.output_dir)




if __name__ == "__main__":
    train()
