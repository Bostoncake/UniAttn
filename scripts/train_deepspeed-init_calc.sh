#!/bin/bash

# command line input: configurations for the Superblock
grouping_idx=$1
grouping_begin_idx=$2
grouping_end_idx=$3
total_layers=$4

TOTAL_INIT_LAYER_INDEX=$(python -c "import math; a=$grouping_end_idx; b=$grouping_begin_idx; c=$grouping_idx; print(math.floor((a-b)/c)*(c-1)-1)")

# TODO: modify those fields
DATA_DIR=path/to/sampled/data   # e.g., dataset/pmc_llama_instructions_1000.jsonl
MODEL_DIR=path/to/your/model
if [[ $DATA_DIR == *"tulu3"* ]]; then
    dataname=tulu
elif [[ $DATA_DIR == *"pmc"* ]]; then
    dataname=pmc
fi


if [[ $MODEL_DIR == *"Llama-2"* ]]; then
    model_name=llama2
elif [[ $MODEL_DIR == *"Llama-3.1"* ]]; then
    model_name=llama3
elif [[ $MODEL_DIR == *"Mistral"* ]]; then
    model_name=mistral_7b
elif [[ $MODEL_DIR == *"gemma-2"* ]]; then
    model_name=gemma2_9b
fi

save_dir=valid_weight_${dataname}/${model_name}
mkdir -p ${save_dir}

# calculate the statistics on the plain base model
CUDA_VISIBLE_DEVICES=0 python utils/initialization.py \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_DIR} \
    --bf16 True \
    --grouping_idx ${grouping_idx} \
    --grouping_begin_idx ${grouping_begin_idx} \
    --grouping_end_idx ${grouping_end_idx} \
    --save_dir ${save_dir} \
    --mode "base"

# calculate the statistics on the model with softmax unification
for i in $(seq 0 ${TOTAL_INIT_LAYER_INDEX})
do
    CUDA_VISIBLE_DEVICES=0 python utils/initialization.py \
        --model_name_or_path ${MODEL_DIR} \
        --data_path ${DATA_DIR} \
        --bf16 True \
        --mode "softmax_init_sequential" \
        --grouping_idx ${grouping_idx} \
        --grouping_begin_idx ${grouping_begin_idx} \
        --grouping_end_idx ${grouping_end_idx} \
        --use_layer_idx ${i} \
        --save_dir ${save_dir} \
    # 计算init weight，存储
    python utils/calc_init_weights.py ${i} ${save_dir} ${grouping_idx} ${grouping_begin_idx} ${grouping_end_idx} ${total_layers}
done

TOTAL_INIT_LAYERS=$(python -c "import math; a=$grouping_end_idx; b=$grouping_begin_idx; c=$grouping_idx; print(math.floor((a-b)/c)*(c-1))")
python utils/concat_init_weights.py ${TOTAL_INIT_LAYERS} ${save_dir}
