#!/bin/bash

# ========= Hyperparameters =========
max_steps=10
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=3
sample_clients=3
lora_r=32
lora_alpha=64         # Twice of lora_r
lr=5e-5

# ========= Model =========
# model_name_or_path="meta-llama/Llama-2-7b-hf"
model_name_or_path="meta-llama/Llama-2-13b-hf"
output_dir="./output"

# ========= FL settings =========
gpu=0
fed_alg="fedavg"

# ========= Dataset Selection =========
# ---- Dataset 1: Medical Meadow Flashcards ----
# dataset_name="medalpaca/medical_meadow_medical_flashcards"
# dataset_sample=33955

# ---- Dataset 2: MathInstruct ----
# dataset_name="TIGER-Lab/MathInstruct"
# dataset_sample=262039

# ---- Dataset 3: WizardLM 70k ----
# dataset_name="WizardLMTeam/WizardLM_evol_instruct_70k"
# dataset_sample=70000

# ---- Dataset 4: Alpaca GPT-4 (default) ----
dataset_name="vicgalle/alpaca-gpt4"
dataset_sample=52002

# ---- Dataset 5: FinGPT (alternative) ----
# dataset_name="FinGPT/fingpt-sentiment-train"
# dataset_sample=76772

# ---- Dataset 6: Local data ----
# local_data_dir=""     # Uncomment if using local dataset
# dataset_name="your_local_dataset"
# dataset_sample=20000

# ========= Run training =========
CUDA_VISIBLE_DEVICES=$gpu python fedshield_llm.py \
    --learning_rate $lr \
    --model_name_or_path $model_name_or_path \
    --dataset_name $dataset_name \
    --dataset_sample $dataset_sample \
    --fed_alg $fed_alg \
    --num_clients $num_clients \
    --sample_clients $sample_clients \
    --max_steps $max_steps \
    --num_rounds $num_rounds \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --seq_length $seq_length \
    --peft_lora_r $lora_r \
    --peft_lora_alpha $lora_alpha \
    --use_peft \
    --load_in_8bit \
    --output_dir $output_dir \
    --template "alpaca"  # Keep "alpaca" uncommented for training
