"""
Usage:
python merge_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH]
"""
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_lora(base_model_name, lora_path):
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=True)
    peft_model = PeftModel.from_pretrained(base_model, lora_path, local_files_only=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # Merge LoRA weights into the base model and unload
    model = peft_model.merge_and_unload()

    # Save merged model and tokenizer
    target_model_path = lora_path.replace("checkpoint", "full")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)

    print(f"Merged model saved to {target_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights into the base model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model or Hugging Face model ID.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint.")

    args = parser.parse_args()

    # Execute merge
    merge_lora(args.base_model_path, args.lora_path)
