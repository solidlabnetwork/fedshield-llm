# =======================
# ENVIRONMENT SETUP
# =======================
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# =======================
# MERGE LoRA WITH BASE MODEL
# =======================
python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path path/to/your/fingpt-sentiment-checkpoint-200

# =======================
# GENERATE MODEL RESPONSES
# =======================
python eval_utils/generate_answer.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path path/to/your/alpaca-gpt4-alt-checkpoint-200

python eval_utils/generate_answer.py \
  --base_model_path path/to/your/full-model-checkpoint-200

