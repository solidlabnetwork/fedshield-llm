
# =======================
# ENVIRONMENT SETUP
# =======================
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# =======================
# MERGE LoRA WITH BASE MODEL
# =======================
python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /disk/solidlab-server/lclhome/mmia001/FLS/output/fingpt-sentiment-train_20000_fedavg_c3s3_i10_b16a1_l512_r32a64_20241117220133/checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /disk/solidlab-server/lclhome/mmia001/FLS/output/fingpt-sentiment-train_76772_fedavg_c3s3_i10_b16a1_l512_r32a64_20241118163956/checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /disk/solidlab-server/lclhome/mmia001/FLS/output/alpaca-gpt4_52002_fedavg_c3s3_i10_b16a1_l512_r32a64_20241119105258/checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/alpaca-gpt4_52002_fedavg_c3s3_i10_b16a1_l512_r32a64_20250124223240/checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/WizardLM_evol_instruct_70k_70000_fedavg_c3s3_i10_b16a1_l512_r32a64_20250117230210/checkpoint-200

python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/WizardLM_evol_instruct_70k_70000_fedavg_c3s3_i10_b16a1_l512_r32a64_20250118020537/checkpoint-200

# =======================
# GENERATE MODEL RESPONSES
# =======================
python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/alpaca-gpt4_52002_fedavg_c3s3_i10_b16a1_l512_r32a64_20250124223240/checkpoint-200

python gen_model_answer_mt.py \
  --base_model_path /disk/solidlab-server/lclhome/mmia001/FLS/output/alpaca-gpt4_52002_fedavg_c3s3_i10_b16a1_l512_r32a64_20241119105258/full-200

python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/WizardLM_evol_instruct_70k_70000_fedavg_c3s3_i10_b16a1_l512_r32a64_20250117230210/checkpoint-200


python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/MathInstruct_262039_fedavg_c3s3_i10_b16a1_l512_r32a64_20250120195709/checkpoint-200


python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/alpaca-gpt4_52002_fedavg_c3s3_i10_b16a1_l512_r32a64_20250327103422/checkpoint-200

python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /disk/solidlab-server/lclhome/mmia001/hellm/output/alpaca-gpt4_52002_fedavg_c3s3_i10_b16a1_l512_r32a64_20250327103422/checkpoint-200

# =======================
# OTHER UTILITIES
# =======================

python evaluation/open_ended/gen_model_answer_mt.py \
  --base_model_path "meta-llama/Llama-2-13b-hf" \
  --template alpaca \
  --lora_path output/alpaca-gpt4_52002_fedavg_c3s3_i10_b16a1_l512_r32a64_20250502175443/checkpoint-200
