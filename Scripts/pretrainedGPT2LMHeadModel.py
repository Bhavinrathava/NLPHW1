from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer():
    # Load the pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer

def read_data_wiki2test(file_path):
    # Read the test data from a file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def encode_data(tokenizer, data, max_length=1024):
    # Encode the data and split into chunks of max_length   
    chunks = [data[i:i + max_length] for i in range(0, len(data), max_length)]
    encoded_chunks = [tokenizer.encode(chunk, return_tensors='pt') for chunk in chunks]
    return encoded_chunks

def calculate_perplexity_wiki2test(model, encoded_chunks):
    # Calculate the perplexity for each chunk and return the average
    log_probs = []
    with torch.no_grad():
        for chunk in tqdm(encoded_chunks, desc='Calculating perplexity for wiki2test'):
            inputs = torch.tensor(chunk).unsqueeze(0)  # Add batch dimension
            outputs = model(inputs, labels=inputs)
            log_probs.append(outputs.loss.item())  # Collect log probability for each chunk
    
    # Convert log probabilities to perplexity
    perplexities = [torch.exp(torch.tensor(lp)).item() for lp in log_probs]
    return sum(perplexities) / len(perplexities)  # Return average perplexity

def wiki2testGPTLMHeadModel():
    model, tokenizer = load_model_and_tokenizer() # Load the pre-trained GPT-2 model and tokenizer
    data = read_data_wiki2test('Data/wiki2.test.txt') # Read the test data from a file
    encoded_chunks = encode_data(tokenizer, data)   # Encode the data and split into chunks of max_length = 1024
    avg_perplexity = calculate_perplexity_wiki2test(model, encoded_chunks) # Calculate the perplexity for each chunk and return the average
    print(f'Average Perplexity for wiki2test: {avg_perplexity}') # Print the average perplexity

if __name__ == "__main__":
    wiki2testGPTLMHeadModel()
