import copy
import os
import pandas as pd
import numpy as np
import torch
import tenseal as ts
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

from utils import *
from utils.HE_utils import *
from utils.prune_utils import *
from fl_utils import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Load config and arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Create encryption context =====
context = ts.context(
    ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
for client_idx, client_data in enumerate(local_datasets):
    print(f"Client {client_idx + 1} has {len(client_data)} samples.")

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.config.use_cache = False  # Silence the warnings. Re-enable for inference!

if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name_or_path, use_fast=False, padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token  # Following vicuna

# ===== Define the formatting function =====
formatting_prompts_func, response_template = get_formatting_prompts_func(
    script_args.template, tokenizer.eos_token
)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for _ in range(fed_args.num_clients)]
output_file = os.path.join(script_args.output_dir, "HE_training_loss.csv")

if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round"] + [f"Client_{i+1}" for i in range(fed_args.num_clients)])

for round in range(fed_args.num_rounds):
    global_state_dict = global_dict
    local_models = []
    clients_this_round = get_clients_this_round(fed_args, round)

    # Use pruning args from config
    pruning_start = script_args.pruning_start_round
    pruning_end = script_args.pruning_end_round
    pruning_target = script_args.pruning_target
    pruning_initial = script_args.pruning_initial

    pruning_rate = max(
        0,
        (round - pruning_start) / (pruning_end - pruning_start)
    ) * (pruning_target - pruning_initial) + pruning_initial

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    print(f"Calculated pruning_rate: {pruning_rate:.4f}")

    round_losses = []

    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            round_losses.append(-1)
            continue

        set_peft_model_state_dict(model, global_state_dict)

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)

        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
        training_args = get_training_args(script_args, new_lr)

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

        local_dict = get_peft_model_state_dict(model)

        if round >= pruning_start:
            print("Pruning applied in this round")
            print("Pruning rate:", pruning_rate)
            mask_dict = Prune_LLM(local_dict, sparsity=pruning_rate)
            pruned_local_state_dict = set_weight_by_mask(local_dict, mask_dict)
            enc_local_dict = encrypt_weights(pruned_local_state_dict, context)
        else:
            print("No pruning applied in this round")
            enc_local_dict = encrypt_weights(local_dict, context)

        local_models.append(enc_local_dict)

    # Aggregate client models using FedAvg
    agg_dict = aggregate_encrypted_updates(local_models, context)
    dec_update = decrypt_weights(agg_dict, context, global_state_dict)

    global_dict = dec_update
    set_peft_model_state_dict(model, global_dict)

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([round+1] + round_losses)

    if (round + 1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}"))

np.save(
    os.path.join(script_args.output_dir, "training_loss.npy"),
    np.array(training_loss),
)
