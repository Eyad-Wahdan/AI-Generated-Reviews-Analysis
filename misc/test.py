import transformers
import torch
import torch.nn.functional as F
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = transformers.AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
tokenizer = transformers.AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
detector.eval()
detector.to(device)
Text_input=["I'm not a chatbot"]
with torch.no_grad(): 
  inputs = tokenizer(Text_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
  inputs = {k:v.to(device) for k,v in inputs.items()}
  output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
  print("Probability of AI-generated texts is",output_probs)

df = pd.read_csv('2023-sample.csv')

print(df.head())

max_length = 512

# df = df[df['text'].apply(lambda x: isinstance(x, str))]

# df = df[df['text'].apply(lambda x: len(tokenizer(x, truncation=False)['input_ids']) <= max_length)]

with torch.no_grad():
    ai_prob = []
    for text in df['text']:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
        ai_prob.append(output_probs[0])
        progress = len(ai_prob) / len(df) * 100
        print(f'{progress:.2f}%', end='\r')
    df['ai_prob'] = ai_prob

df.to_csv('2023_ai_prob.csv')

print(f'{df["ai_prob"].mean():,}')
print(f'{df["ai_prob"].sum():,}')
print(f'{df["ai_prob"].max():,}')
print(f'{df["ai_prob"].min():,}')
print(f'{df["ai_prob"].median():,}')
print(f'{df["ai_prob"].std():,}')
print(f'{df["ai_prob"].quantile(0.25):,}')
print(f'{df["ai_prob"].quantile(0.75):,}')
print(f'{df["ai_prob"].quantile(0.95):,}')