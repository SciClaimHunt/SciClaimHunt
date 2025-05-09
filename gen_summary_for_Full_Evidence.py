import os
import pandas as pd
import gc
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset for text summarization/paraphrasing tasks."""
    def __init__(self, df, text_column, prompt_template=None):
        self.df = df
        self.text_column = text_column
        self.prompt_template = prompt_template or "Could you rephrase the given piece of text and generate a continous long well balanced summary containing all the important parts of the text snippet. The text might be broken or incomplete. Try to think clearly and generate the summary. Every Answer should begin with the Word Summary. The text given is {text}."
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        user_content = self.df.iloc[idx][self.text_column]
        prompt = self.prompt_template.format(text=user_content)
        return prompt

def clean_text(text, keep_spaces=True):
    """Clean text by removing non-alphanumeric characters."""
    if keep_spaces:
        return re.sub(r'[^\s\w]', '', text)
    return re.sub(r'[^\w]', '', text)

def load_model_and_tokenizer(model_id, cache_dir, load_in_4bit=True, max_length=4196):
    """Load model and tokenizer with specified configuration."""
    logger.info(f"Loading model: {model_id}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        cache_dir=cache_dir,
        max_length=max_length,
        device_map='auto'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    return model, tokenizer

def generate_summaries(model, tokenizer, dataloader, max_new_tokens=512, input_length_threshold=3500):
    """Generate summaries using the model."""
    summaries = []
    original_texts = []
    
    for batch in tqdm(dataloader, desc='Generating Summaries'):
        enc = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True, truncation=True)
        
        # Adjust generation based on input length
        if enc['input_ids'].shape[1] <= input_length_threshold:
            outputs = model.generate(
                enc['input_ids'].to('cuda'),
                max_new_tokens=max_new_tokens
            )
        else:
            outputs = model.generate(
                enc['input_ids'].to('cuda'),
                max_new_tokens=max_new_tokens
            )
            
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(decoded_outputs)
        original_texts.extend(batch)
    
    return summaries, original_texts

def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, 
        args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        max_length=args.max_length
    )
    
    # Load and prepare dataset
    logger.info(f"Loading dataset from: {args.dataset_path}")
    dataset = pd.read_csv(args.dataset_path)
    
    # Apply optional dataset filtering
    if args.start_idx is not None and args.end_idx is not None:
        dataset = dataset.iloc[args.start_idx:args.end_idx]
    elif args.start_idx is not None:
        dataset = dataset.iloc[args.start_idx:]
    
    if args.drop_na:
        dataset = dataset.dropna(subset=[args.text_column])
    
    # Create dataset and dataloader
    text_dataset = TextDataset(
        dataset, 
        text_column=args.text_column,
        prompt_template=args.prompt_template
    )
    
    dataloader = DataLoader(
        text_dataset, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle
    )
    
    # Generate summaries
    summaries, original_texts = generate_summaries(
        model, 
        tokenizer, 
        dataloader,
        max_new_tokens=args.max_new_tokens,
        input_length_threshold=args.input_length_threshold
    )
    
    # Save results
    dataset[args.output_column] = summaries
    logger.info(f"Saving results to: {args.output_path}")
    dataset.to_csv(args.output_path, index=args.save_index)
    
    # Free memory
    if args.free_memory:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Summarization/Paraphrasing with LLMs')
    
    # Model configuration
    parser.add_argument('--model_id', type=str, default='google/gemma-2-2b-it',
                        help='Model ID to load from Hugging Face')
    parser.add_argument('--cache_dir', type=str, default='/scratch/cmodels/gemma',
                        help='Directory to cache the downloaded model')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Whether to load the model in 4-bit quantization')
    parser.add_argument('--max_length', type=int, default=4196,
                        help='Maximum sequence length for the model')
    
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset CSV file')
    parser.add_argument('--text_column', type=str, default='retrievedsentences',
                        help='Column name containing the text to summarize')
    parser.add_argument('--drop_na', action='store_true',
                        help='Drop rows with NA values in the text column')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Starting index for dataset slicing')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index for dataset slicing')
    
    # Processing configuration
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the dataset')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--input_length_threshold', type=int, default=3500,
                        help='Threshold for input length handling')
    parser.add_argument('--prompt_template', type=str, default=None,
                        help='Custom prompt template with {text} placeholder')
    
    # Output configuration
    parser.add_argument('--output_path', type=str, default='./summarized_output.csv',
                        help='Path to save the output CSV file')
    parser.add_argument('--output_column', type=str, default='paraphrasedSummary',
                        help='Column name for the generated summaries')
    parser.add_argument('--save_index', action='store_true',
                        help='Whether to save DataFrame index in output CSV')
    
    # Misc
    parser.add_argument('--free_memory', action='store_true',
                        help='Free memory after processing')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Adjust logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    main(args)
