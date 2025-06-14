# 🛡️FedShield-LLM: A Secure and Scalable Federated Fine-Tuned Large Language Model

FedShield-LLM is a novel framework that enables secure and efficient federated fine-tuning of Large Language Models (LLMs) across organizations while preserving data privacy. By combining pruning with Fully Homomorphic Encryption (FHE) for Low-Rank Adaptation (LoRA) parameters, FedShield-LLM allows encrypted computation on model updates, reducing the attack surface and mitigating inference attacks like membership inference and gradient inversion. Designed for **cross-silo federated environments**, the framework optimizes computational and communication efficiency, making it suitable for small and medium-sized organizations.

**Key Features:**
- 🚀 **Encrypted LoRA aggregation** using Fully Homomorphic Encryption (FHE).
- ⚡ **Communication-efficient updates** through aggressive pruning.
- 🛡️ **Privacy-preserving defense** against membership inference and gradient inversion attacks.
- 📈 **Empirically validated**: Outperforms baseline methods while maintaining robust privacy protection.

---
## 🛠️ Requirements

- Python 3.9+
- PyTorch 2.1.2
- TenSEAL
- Torchvision
- CUDA 11.8 (for GPU support)

## 📚 Datasets

This project supports various datasets for federated fine-tuning:

- **Medical Meadow Flashcards:** [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) (33,955 samples)
- **MathInstruct:** [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) (262,039 samples)
- **WizardLM 70k:** [WizardLMTeam/WizardLM_evol_instruct_70k](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k) (70,000 samples)
- **Alpaca GPT-4:** [vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) (52,002 samples)
- **FinGPT:** [FinGPT/fingpt-sentiment-train](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) (76,772 samples)


## 🤖 Models

Supported base models for federated fine-tuning:

- **LLaMA-2 7B:** [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **LLaMA-2 13B:** [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)


## 📦 Environment Setup

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🔧Finetuning using FedShield-LLM
```bash
bash run_scripts/fed.sh
```
## 🔧Finetuning using Vanilla-FL
```bash
bash run_scripts/vanilla.sh
```

## 🔧 Merge LoRA with Base Model

To merge the LoRA weights with the base model, use the following command:

```bash
python utils/merge_lora.py --base_model_path "meta-llama/Llama-2-7b-hf" \
  --lora_path /path/to/your/lora/checkpoint
```

## 📝 Generate Answers Using Merged Model

To generate answers using the merged model, use the following command:

```bash
python eval_utils/generate_answer.py \
  --base_model_path /path/to/your/merged/model/checkpoint
```
## 📝 Generate Answers Using Base Model + LoRA Weights

To generate answers using the base model and LoRA weights, use the following command:

```bash
python eval_utils/generate_answer.py \
  --base_model_path "meta-llama/Llama-2-7b-hf" \
  --template alpaca \
  --lora_path /path/to/your/lora/checkpoint
```

## 📚 Citation

If you use this project or codebase in your research or publication, please cite it as follows:
```bash
@article{mia2025fedshield,
  title={FedShield-LLM: A Secure and Scalable Federated Fine-Tuned Large Language Model},
  author={Mia, Md Jueal and Amini, M Hadi},
  journal={arXiv preprint arXiv:2506.05640},
  year={2025}
}
```

## Acknowledgements

This repository is based on [**OpenFedLLM**](https://github.com/rui-ye/OpenFedLLM) thanks to the original authors for their works!
