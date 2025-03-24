# read 2023-sample-long-zerogpt-result.csv and replace all \n with \\n

file = None

with open('2023-sample-long-zerogpt-result.csv', 'r') as f:
    file = f.read()

file = file.replace('\n\n', '\\n\\n')

with open('2023-sample-long-zerogpt-result.csv', 'w') as f:
    f.write(file)