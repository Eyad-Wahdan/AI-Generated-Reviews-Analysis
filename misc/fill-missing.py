import pandas as pd
import requests
import json
import csv

df = pd.read_csv('2023-sample-long-zerogpt-result.csv', quotechar='"', encoding='utf-8', quoting=csv.QUOTE_ALL)
df['text'] = df['text'].str.replace('\\n', '\n')

url = 'https://api.zerogpt.com'


def login():
    response = requests.post(url + '/api/auth/login',
                             json={'email': 'midall01.md@gmail.com', 'password': 'E4VJ9D7i@Tux@iw*uh.RuMyqmo4U!B'})
    return response.json()['data']['token']


def detectText(token: str, text: str):
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}',
               'ApiKey': '7256620d-8f12-4393-9f32-d9920e24bde3'}
    body = {'input_text': text}
    response = requests.post(url + '/api/detect/detectText', json=body, headers=headers)
    print(response.text)
    return response.text


token = login()

print(token)
print(f'Bearer {token}')

# df['ai_likelihood'] = df['text'].apply(lambda x: detectText(token, x))

# ai_likelihood is a json string

for index, row in df.iterrows():
    if json.loads(row['ai_likelihood'])['data'] is None:
        print(f'Row {index} has null ai_likelihood')
        df.at[index, 'ai_likelihood'] = detectText(token, row['text'])

print(df.head())

df['text'] = df['text'].str.replace('\n', '\\n')

df.to_csv('2023-sample-long-zerogpt-result.csv', index=False, quotechar='"', encoding='utf-8', escapechar='\\',
          quoting=csv.QUOTE_ALL)