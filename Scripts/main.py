import transformers as tr
from nltk.tokenize import WordPunctTokenizer as wpt
import math as m
from collections import Counter
import sys
from pretrainedGPT2LMHeadModel import wiki2testGPTLMHeadModel
from examplesPerplexityGPTLMHeadMOdel import examplesGPTLMHeadModel


def main():
    #sampleTokenizerTest()
    data = readData('../Data/wiki2.train.txt')
    testData = readData('../Data/wiki2.test.txt')
    
    data = preprocessData(data)
    testData = preprocessData(testData)

    trainTokens = tokenizeWithWordPunctTokenizer(data)
    testTokens = tokenizeWithWordPunctTokenizer(testData)

    trainGPT2Tokens = tokenizeWithGPT2FastTokenizer(data)
    testGPT2Tokens = tokenizeWithGPT2FastTokenizer(testData)

    for n in [1,2,3,7]:
        perplexity = calculatePerplexityForNGram(trainTokens, testTokens, n)
        print(f'Perplexity of the model using WordPunctTokenizer and {n}-grams: {perplexity}')

        perplexity = calculatePerplexityForNGram(trainGPT2Tokens, testGPT2Tokens, n)
        print(f'Perplexity of the model using GPT2FastTokenizer and {n}-grams: {perplexity}')
    pass

    wiki2testGPTLMHeadModel()
    examplesGPTLMHeadModel()

def preprocessData(data):
    # Preprocess the data
    # Input: data (string)
    # Output: data (string)
    
    # Remove the newline characters
    data = data.replace('\n', ' ')
    
    # Remove the extra spaces
    data = ' '.join(data.split())
    
    # Remove special characters but keep the punctuations
    data = ''.join(e for e in data if e.isalnum() or e in [' ', '.', ',', '!', '?', ':', ';', '-'])

    # Convert the data to lowercase
    data = data.lower()

    return data

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

    # Split the data into 1024 tokens at a time
    # tokens = []
    # for i in range(0, len(data), 1024):
    #     tokens.extend(tokenizer.tokenize(data[i:i+1024]))

    # gptTokens= []
    # for temp in tokens:
    #     gptTokens.extend(tokenizer.tokenize(temp))
    tokens = tokenizer.tokenize(data)
    return tokens

def calculateNGrams(tokens, n):
    # Calculate the n-grams from the tokens
    # Input: tokens (list of strings), n (int)
    # Output: ngrams (list of strings)
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def calculateNGramProbabilities(ngrams, n):
    # Calculate the probabilities of the n-grams
    # Input: ngrams (list of strings)
    # Output: ngramProbabilities (dictionary of strings to floats)
    
    nGramCounts = Counter(ngrams)
    probabilities = {}
    
    if (n == 1):
        total = sum(nGramCounts.values())
        for ngram in nGramCounts:
            probabilities[ngram] = nGramCounts[ngram] / total
    else:
        n1Grams = [' '.join(ngram.split()[:-1] ) for ngram in ngrams]
        n1GramCounts = Counter(n1Grams)
        unique_words = set(word for ngram in nGramCounts for word in ngram)
        v = len(unique_words)
        for ngram in nGramCounts:
            n1gram = ' '.join(ngram.split()[:-1] )
            probabilities[ngram] = (1 + nGramCounts[ngram]) / (n1GramCounts[n1gram] + v)

    return probabilities

def calculatePerplexity(nGrams, probabilities):
    # Calculate the perplexity of the model
    # Input: nGrams (list of strings), probabilities (dictionary of strings to floats)
    # Output: perplexity (float)
    
    logProb = 0
    for ngram in nGrams:
        if ngram in probabilities:
            logProb += m.log(probabilities[ngram])
        else:
            logProb += m.log(1/len(nGrams))
    perplexity = m.exp(-logProb / len(nGrams))

    return perplexity

def calculatePerplexityForNGram(dataTokens, testDataTokens, n = 2):
    # Calculate the perplexity of the model using n-grams
    # Input: data (string), testData (string)
    # Output: perplexity (float)
    
    # genrate N Grams
    ngrams = calculateNGrams(dataTokens, n)
    testNGrams = calculateNGrams(testDataTokens, n)

    # Calculate the probabilities of the n-grams
    ngramProbabilities = calculateNGramProbabilities(ngrams, n)

    # Calculate the perplexity
    perplexity = calculatePerplexity(testNGrams, ngramProbabilities)
    return perplexity


    


if __name__ == '__main__':
    main()
