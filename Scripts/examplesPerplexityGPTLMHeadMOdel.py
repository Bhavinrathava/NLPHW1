from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    return model, tokenizer

def read_examples(file_path):
    examples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                examples.append(line.strip())
    return examples

def calculate_perplexity(model, tokenizer, text):
    inputs = torch.tensor([tokenizer.encode(text, truncation=True, max_length=1024)]).to(model.device)
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
    return torch.exp(loss).item()

def examplesGPTLMHeadModel():
    model, tokenizer = load_model_and_tokenizer()
    examples = read_examples('../Data/examples.txt')
    
    for idx, example in enumerate(tqdm(examples, desc='Processing examples')):
        perplexity = calculate_perplexity(model, tokenizer, example)
        print(f'Example {idx + 1}: Perplexity using GPTLNHeadModel = {perplexity}')

if __name__ == "__main__":
    examplesGPTLMHeadModel()



