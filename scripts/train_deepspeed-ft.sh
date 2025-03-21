#!/bin/bash

# TODO: modify those fields
DATA_DIR=path/to/ft/data
MODEL_DIR=path/to/your/model

MODEL_MODE="plain"

if [[ $MODEL_DIR == *"Llama-2"* ]]; then
    MODEL_NAME_ALIAS=llama2_7b
    DECODER_LAYER=LlamaDecoderLayer
    lr=2e-5
elif [[ $MODEL_DIR == *"llama2"* ]]; then
    MODEL_NAME_ALIAS=llama2_7b
    DECODER_LAYER=LlamaDecoderLayer
    lr=2e-5
elif [[ $MODEL_DIR == *"Llama-3.1"* ]]; then
    MODEL_NAME_ALIAS=llama3.1_8b
    DECODER_LAYER=LlamaDecoderLayer
    lr=7e-6
elif [[ $MODEL_DIR == *"Mistral"* ]]; then
    MODEL_NAME_ALIAS=mistral_7b
    DECODER_LAYER=MistralDecoderLayer
    lr=1e-6
elif [[ $MODEL_DIR == *"gemma-2"* ]]; then
    MODEL_NAME_ALIAS=gemma2_9b
    DECODER_LAYER=Gemma2DecoderLayer
    lr=1e-6
else
    MODEL_NAME_ALIAS=NOT_IMPLEMENTED
    DECODER_LAYER=NOT_IMPLEMENTED
    lr=NOT_IMPLEMENTED
fi
DATETIME=$(date +%m%d)
OUTPUT_DIR=output/pmc_output_${MODEL_NAME_ALIAS}_${lr}_${DATETIME}
mkdir -p ${OUTPUT_DIR}


torchrun --nproc_per_node=8 --master_port=54321 train.py \
    --model_name_or_path ${MODEL_DIR} \
    --data_path ${DATA_DIR} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --model_max_length 1024 \
    --save_total_limit 100 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --report_to "none" \
    --fsdp_transformer_layer_cls_to_wrap ${DECODER_LAYER} \
    --tf32 True \
    --model_mode ${MODEL_MODE}