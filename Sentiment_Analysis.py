import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def Sentiment(tokenizer, model, pharse):
    tokens = tokenizer.encode(pharse, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))

