import pandas as pd
import transformers
import torch
import langdetect

def is_english(text):
    try:
        return langdetect.detect(text) == 'en'
    except:
        return False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = transformers.AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
tokenizer = transformers.AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")

df = pd.read_csv('2023.csv')

print(df.head())

max_length = 512

df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x) > 0)]

df['token_count'] = df['text'].apply(lambda x: len(tokenizer(x, truncation=False)['input_ids']))

df = df[df['token_count'] <= max_length]

print(len(df))

df = df.sort_values(by='token_count', ascending=False)
print(len(df))

print('start filtering language')

df = df.head(20_000)
print(len(df))

df = df[df['text'].apply(lambda x: is_english(x))]
print(len(df))

df = df.head(1_000)

print(df.head())

df.to_csv('2023-sample-long.csv', index=False)