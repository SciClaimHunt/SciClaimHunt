import os
import gc
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
    pipeline
)
import bitsandbytes as bnb
from datasets import set_caching_enabled

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
set_caching_enabled(False)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def generate_prompt(data_point):
    return f"""You are an expert in making judgements. You are given a claim and some sentences. Your job is to verify that the evidence encapsulated in retrieved sentences support the claim or not. The retrieved sentences come from a research paper. If the evidence encapsulated in the retrieved sentences support the claim then you need to answer as 'positive' otherwise 'negative',
retrieved sentences: {data_point['retrievedsentences']},
claim: {data_point['Claim']},
label: {data_point['Type']}"""

def generate_test_prompt(data_point):
    return f"""You are an expert in making judgements. You are given a claim and some sentences. Your job is to verify that the evidence encapsulated in retrieved sentences support the claim or not. The retrieved sentences come from a research paper. If the evidence encapsulated in the retrieved sentences support the claim then you need to answer as 'positive' otherwise 'negative',
retrieved sentences: {data_point['retrievedsentences']},
claim: {data_point['Claim']},
label: """

def predict(test, model, tokenizer):
    y_pred = []
    answer_gen = []
    categories = ['negative', 'positive']

    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=2, temperature=0.1)
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()
        answer_gen.append(answer.lower())
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")
    return y_pred, answer_gen

def evaluate(y_true, y_pred):
    categories = ['negative', 'positive']
    mapping = {label: idx for idx, label in enumerate(categories)}

    y_true_mapped = np.vectorize(mapping.get)(y_true)
    y_pred_mapped = np.vectorize(mapping.get)(y_pred)

    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')

    for label, name in enumerate(categories):
        indices = [i for i, v in enumerate(y_true_mapped) if v == label]
        acc = accuracy_score([y_true_mapped[i] for i in indices], [y_pred_mapped[i] for i in indices])
        print(f'Accuracy for label {name}: {acc:.3f}')

    print('\nClassification Report:')
    print(classification_report(y_true_mapped, y_pred_mapped, target_names=categories))

    cm = confusion_matrix(y_true_mapped, y_pred_mapped)
    pd.DataFrame(cm).to_csv('./confusion_report')
    print('\nConfusion Matrix:')
    print(cm)

def main(args):
    df = pd.read_csv(args.dataset_csv_file_loc)
    df = df.fillna('')
    df['label'] = df['Type'].apply(lambda x: 0 if x == 'positive' else 1)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df, dev_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df['text'] = train_df.apply(generate_prompt, axis=1)
    dev_df['text'] = dev_df.apply(generate_prompt, axis=1)

    test_labels = test_df['Type']
    test_df = pd.DataFrame(test_df.apply(generate_test_prompt, axis=1), columns=["text"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_data = Dataset.from_pandas(train_df[["text"]])
    eval_data = Dataset.from_pandas(dev_df[["text"]])

    modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=False,
        lr_scheduler_type="cosine",
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=0.2
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=3072,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )

    torch.cuda.empty_cache()
    gc.collect()

    trainer.train()
    model.config.use_cache = True
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    y_pred, text_predicted = predict(test_df, model, tokenizer)
    evaluate(test_labels, y_pred)
    test_df['true_labels'] = test_labels
    test_df['predicted_labels'] = text_predicted
    test_df.to_csv('./new_csv', index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate LLaMA PEFT model')

    parser.add_argument('--dataset_csv_file_loc', type=str, default='/scratch/fourccmodels/csv_file_new',
                        help='CSV file with training data')
    parser.add_argument('--output_dir', type=str, default='/scratch/cmodels/saved_model_ckps',
                        help='Directory to save trained model and tokenizer')
    parser.add_argument('--base_model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Base model name or path')
    parser.add_argument('--cache_dir', type=str, default='/scratch/cmodels/llama3.2',
                        help='Directory for model cache')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Per-device training batch size')

    args = parser.parse_args()
    main(args)
