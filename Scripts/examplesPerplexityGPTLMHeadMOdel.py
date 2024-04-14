from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from main import readData, preprocessData, tokenizeWithWordPunctTokenizer, tokenizeWithGPT2FastTokenizer, calculatePerplexityForNGram

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
    examples = read_examples('Data/examples.txt')
    data = readData('Data/wiki2.train.txt')

    for idx, example in enumerate(examples):
        perplexity = calculate_perplexity(model, tokenizer, example)

        print(f'Example {idx + 1}: \n Example Text : {example} \n Perplexity using GPTLNHeadModel = {perplexity}')
        testData = example
        
        data = preprocessData(data)
        testData = preprocessData(testData)

        trainTokens = tokenizeWithWordPunctTokenizer(data)
        testTokens = tokenizeWithWordPunctTokenizer(testData)

        for n in [1,2,3,7]:
            if(len(testTokens) < n):
                print(f'Cannot calculate perplexity for {n}-grams as the test data is too short')
                continue
            perplexity = calculatePerplexityForNGram(trainTokens, testTokens, n)
            print(f'Perplexity of the model using WordPunctTokenizer and {n}-grams: {perplexity}')

        print('\n')

if __name__ == "__main__":
    examplesGPTLMHeadModel()



