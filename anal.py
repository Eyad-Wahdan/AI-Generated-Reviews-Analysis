import pandas as pd
import requests
import time
import csv

df = pd.read_csv('2017-sample-long-cleaned.csv', quotechar='"', encoding='utf-8', escapechar='\\')
df['text'] = df['text'].str.replace('\\n', '\n')

url = 'https://api.zerogpt.com'

def login():
    response = requests.post(url + '/api/auth/login', json={'email': 'midall01.md@gmail.com', 'password': 'E4VJ9D7i@Tux@iw*uh.RuMyqmo4U!B'})
    return response.json()['data']['token']

def detectText(token: str, text: str, depth: int = 0):
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}', 'ApiKey': '7256620d-8f12-4393-9f32-d9920e24bde3'}
    body = {'input_text': text}
    response = requests.post(url + '/api/detect/detectText', json=body, headers=headers)
    print(response.text)

    status = response.json()['code']

    if status != 200 and depth < 3:
        print('status code is not 200')
        print('waiting for 10 seconds')

        time.sleep(5 * (depth + 1)) 
        return detectText(token, text, depth + 1)

    return response.text

token = login()

print(token)
print(f'Bearer {token}')

df['ai_likelihood'] = df['text'].apply(lambda x: detectText(token, x))

df['text'] = df['text'].str.replace('\n', '\\n')

print(df.head())

df.to_csv('2017-sample-long-zerogpt-result.csv', index=False, quotechar='"', encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_ALL)