# =======================
# MERGE LoRA WITH BASE MODEL
# =======================
python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /path/to/your/lora/checkpoint
# =======================
# GENERATE MODEL RESPONSES
# =======================
python eval_utils/generate_answer.py \
  --base_model_path /path/to/your/merged/model/checkpoint

python eval_utils/generate_answer.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path /path/to/your/lora/checkpoint

