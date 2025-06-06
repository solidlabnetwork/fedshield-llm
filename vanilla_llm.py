import copy
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from fl_utils import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Load configurations =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)

# ===== Load and process dataset =====
dataset = process_sft_dataset(
    script_args.dataset_name,
    get_dataset(script_args.dataset_name, script_args.local_data_dir),
    script_args.dataset_sample
)
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(data) for data in local_datasets]
for client_idx, num_samples in enumerate(sample_num_list):
    print(f"Client {client_idx + 1} has {num_samples} samples.")

# ===== Load model =====
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=get_model_config(script_args)[1],
    device_map=get_model_config(script_args)[0],
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=get_model_config(script_args)[2],
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.use_cache = False

# ===== Tokenizer and data collator =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Initialize global model state =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
training_loss = [[] for _ in range(fed_args.num_clients)]

# ===== CSV for saving training loss =====
output_file = os.path.join(script_args.output_dir, "Vanilla_training_loss.csv")
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round"] + [f"Client_{i+1}" for i in range(fed_args.num_clients)])

# ===== Federated training =====
for round in tqdm(range(fed_args.num_rounds)):
    print(f">> ==================== Round {round+1} ====================")

    clients_this_round = get_clients_this_round(fed_args, round)
    round_losses = []

    # Local training
    local_models = []
    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            round_losses.append(-1)
            continue

        set_peft_model_state_dict(model, global_dict)
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
        training_args = get_training_args(script_args, new_lr)

        # Local training
        trainer = get_fed_local_sft_trainer(
            script_args=script_args,
            fed_args=fed_args,
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        round_losses.append(results.training_loss)
        local_models.append(get_peft_model_state_dict(model))

    # Aggregation
    aggregated_state_dict = {
        key: torch.stack([local_model[key] for local_model in local_models]).mean(0)
        for key in global_dict.keys()
    }

    global_dict = aggregated_state_dict
    set_peft_model_state_dict(model, global_dict)

    # Save round losses
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([round + 1] + round_losses)

    # Save model checkpoint
    if (round + 1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

# ===== Save final training loss array =====
np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
