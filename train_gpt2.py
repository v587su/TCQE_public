import os
from model.gpt2_dataset import GPT2Dataset
from transformers import GPT2TokenizerFast,DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
from utils import arg_parser

def get_latest_checkpoint(dir_path):
    steps = []
    for file in os.listdir(dir_path):
        if file.startswith('checkpoint'):
            steps.append(int(file.split('-')[-1]))
    if len(steps) > 0:
        return f'checkpoint-{max(steps)}'
    else:
        return None

if __name__ == '__main__':
    args = arg_parser()
    tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
    dataset = GPT2Dataset(args.data_path, tokenizer, is_dev=args.is_dev, max_pos_length=args.text_length, mode=args.mode, language=args.language)
    model = AutoModelForCausalLM.from_pretrained(args.cache_path)
    output_path = f'./results/{args.language}_{args.run_name}'
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="no",
        learning_rate=2e-4,
        save_strategy='epoch',
        save_total_limit=10,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.dataset["train"],
        eval_dataset=dataset.dataset["test"],
    )

    if os.path.exists(output_path) and get_latest_checkpoint(output_path):
        trainer.train(os.path.join(output_path, get_latest_checkpoint(output_path)))
    else:
        trainer.train()
