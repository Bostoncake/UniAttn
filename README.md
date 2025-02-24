# UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs
Official PyTorch implementation of UniAttn, an LLM post-training method for fine-tuning domain-specific or general-performance-boosted instruction following LLMs that enjoy memory efficiency and inference acceleration. 

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Post-training is essential for adapting Large Language Models (LLMs) to real-world applications. Deploying post-trained models faces significant challenges due to substantial memory overhead and noticeable inference latency. Existing work has identified significant redundancies in LLMs and proposed efficient architectures, namely intra-layer KV sharing and cross-layer KV sharing. However, intra-layer KV sharing still results in high inference costs, while cross-layer KV sharing leads to significant performance degradation. As a result, both methods remain suboptimal for post-training pre-trained LLMs. In this paper, we identify that the \texttt{Softmax} operation is a primary bottleneck for LLM inference and discover that it is actually highly redundant during post-training. We propose Softmax \textbf{Uni}fication in \textbf{Att}e\textbf{n}tion (\textbf{UniAttn}), a novel post-training method that unifies Softmax activations across transformer blocks to reduce LLM inference costs. Additionally, UniAttn adopts a linear projection to compensate for the errors induced by Softmax unification. Experiments show that UniAttn matches the performance of standard post-training while significantly reducing inference costs, outperforming existing efficient architectures during post-training.
</details>

## News

 - 2025-02-24: We release the code for UniAttn.
 - 2024-02-01: We have released the pre-print version of the article on arXiv. Check it out [here](https://www.arxiv.org/abs/2502.00439). 

## Preparation

### Datasets

For the medical domain fine-tuning, we use [pmc_llama_instructions](https://huggingface.co/datasets/axiong/pmc_llama_instructions). You can download and save the dataset via `python dataset/prepare_pmc.py`.

For general supervised fine-tuning, we use [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture). We filter the one-round conversations with non-empty inputs for fine-tuning. Simply run `python dataset/prepare_tulu3.py` for preparing the dataset.

The sampled data for calculating the initialization is provided in `dataset/`.

### Models

We use open-source base LLMs for post-training. The sources of each model are listed as following:

 - LLaMA-3.1 8B: https://huggingface.co/meta-llama/Llama-3.1-8B
 - LLaMA-2 7B: https://huggingface.co/meta-llama/Llama-2-7b
 - Mistral 7B: https://huggingface.co/mistralai/Mistral-7B-v0.1
 - Gemma-2 9B: https://huggingface.co/google/gemma-2-9b

### Environment

We suggest `conda` for building isolated running environments.

```
conda create -n uniattn python=3.12
pip install -r requirements.txt
```

## Training

We provide the training scripts in `scripts/`. For fine-tuning each model variant, refer to the following commands:

 - Directly fine-tuning the base model: 
   ```
   bash scripts/train_deepspeed-ft.sh
   ```
 - Fine-tuning model variant with only Softmax unifications (we use LLaMA-3.1 8B as an example): 
   ```
   bash scripts/train_deepspeed-ft_softmax.sh 4 16 33
   ```
 - UniAttn fine-tuning (we use LLaMA-3.1 8B as an example):
   - First, calculate the initialization weights for the linear compensation matrices:
     ```
     bash scripts/train_deepspeed-init_calc.sh 4 16 33 32
     ```
   - Then, conduct UniAttn fine-tuning with the linear transformations:
     ```
     bash scripts/train_deepspeed-ft_uniattn.sh 4 16 33
     ``` 

## Evaluation

We employ [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for model evaluation. Please refer to the official repository for preparing the evaluation environment. By the time we conduct our experiments, we use `lm_eval==0.4.5`.

## Acknowledgements

Part of the repository is based on [AlpaCare](https://github.com/XZhang97666/AlpaCare), [Stanford_Alpaca](https://github.com/tatsu-lab/stanford_alpaca), and [HuggingFace Transformers](https://github.com/huggingface/transformers). Many thanks for the implementations!

## Citation

If you find our work helpful, please consider citing our article:

```BibTeX
@misc{xiong2025uniattn,
      title={UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs}, 
      author={Yizhe Xiong and Wei Huang and Xin Ye and Hui Chen and Zijia Lin and Haoran Lian and Zhenpeng Su and Jungong Han and Guiguang Ding},
      year={2025},
      eprint={2502.00439},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.00439}, 
}
```