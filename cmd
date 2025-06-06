# =======================
# ENVIRONMENT SETUP
# =======================
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# =======================
# MERGE LoRA WITH BASE MODEL
# =======================
python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/fingpt-sentiment-checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/fingpt-sentiment-alt-checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/alpaca-gpt4-checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/alpaca-gpt4-alt-checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/wizardlm-checkpoint-200-a

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/wizardlm-checkpoint-200-b

# =======================
# GENERATE MODEL RESPONSES
# =======================
python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path path/to/your/alpaca-gpt4-alt-checkpoint-200

python gen_model_answer_mt.py \
  --base_model_path path/to/your/full-model-checkpoint-200

python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path path/to/your/wizardlm-checkpoint-200-a

python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path path/to/your/mathinstruct-checkpoint-200

python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path path/to/your/alpaca-gpt4-checkpoint-200-a

python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/alpaca-gpt4-checkpoint-200-a

# =======================
# OTHER UTILITIES
# =======================
python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-13b-hf" \
  --template alpaca \
  --lora_path path/to/your/alpaca-gpt4-checkpoint-200-final
