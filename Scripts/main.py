import transformers as tr
from nltk.tokenize import WordPunctTokenizer as wpt
import math as m


def main():
    sampleTokenizerTest()
    pass

def sampleTokenizerTest():

    # Read the data from the file
    filenames = ['Data/wiki2.train.txt', 'Data/wiki2.valid.txt', 'Data/wiki2.test.txt']

    for filename in filenames:
        data = readData(filename)
        print(f'Number of characters in {filename}: {len(data)}')

        # Tokenize the data using WordPunctTokenizer
        tokensNLTK = tokenizeWithWordPunctTokenizer(data)
        print(f'Number of tokens in {filename}: {len(tokensNLTK)}')

        # Tokenize the data using GPT2FastTokenizer
        tokensGPT2 = tokenizeWithGPT2FastTokenizer(data)
        print(f'Number of tokens in {filename}: {len(tokensGPT2)}')


        # Show sample 200 tokens from the data
        print(f'Sample 200 tokens from {filename}:')
        print(data[:200])

        print(f'Sample 200 tokens from {filename} using WordPunctTokenizer:')
        print(tokensNLTK[:200])

        print(f'Sample 200 tokens from {filename} using GPT2FastTokenizer:')
        print(tokensGPT2[:200])

def calculatePerplexity(predictions, labels):
    # Calculate the perplexity of the model
    # Input: predictions (list of floats), labels (list of floats)
    # Output: perplexity (float)
    
    # Calculate the cross entropy
    cross_entropy = 0
    for i in range(len(predictions)):
        cross_entropy += labels[i] * m.log(predictions[i])
    cross_entropy = -cross_entropy

    # Calculate the perplexity
    perplexity = m.exp(cross_entropy)
    return perplexity

def readData(filename):
    # Read the data from the file
    # Input: filename (string)
    # Output: data - String of the data in the file
    
    with open(filename, 'r', encoding='utf') as file:
        data = file.read()
    return data

def tokenizeWithWordPunctTokenizer(data):
    # Tokenize the data using WordPunctTokenizer
    # Input: data (string)
    # Output: tokens (list of strings)
    
    tokenizer = wpt()
    tokens = tokenizer.tokenize(data)
    return tokens

def tokenizeWithGPT2FastTokenizer(data):
    # Tokenize the data using GPT2FastTokenizer
    # Input: data (string)
    # Output: tokens (list of strings)
    
    tokenizer = tr.GPT2TokenizerFast.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(data)
    return tokens



    


if __name__ == '__main__':
    main()
