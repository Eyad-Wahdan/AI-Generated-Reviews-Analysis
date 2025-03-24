import transformers
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = transformers.AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
tokenizer = transformers.AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
detector.eval()
detector.to(device)

df = pd.read_csv('2023-sample-long-cleaned.csv')

print(df.head())

df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['token_count'] = df['text'].apply(lambda x: len(tokenizer(x, truncation=False)['input_ids']))

print(f'{df["char_count"].mean():,}')
print(f'{df["char_count"].sum():,}')
print(f'{df["word_count"].mean():,}')
print(f'{df["word_count"].sum():,}')
print(f'{df["token_count"].mean():,}')
print(f'{df["token_count"].sum():,}')