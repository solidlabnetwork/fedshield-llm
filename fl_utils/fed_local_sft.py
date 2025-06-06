import torch
from trl import SFTTrainer

def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator):
    if fed_args.fed_alg == 'fedavg':
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer