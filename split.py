def write_buffer(buffer, filename):
    print('writing', filename)
    with open(filename, 'w') as file:
        for line in buffer:
            file.write(line)

with open('data.jsonl') as file:
    buffer = []
    counter = 0
    for line in file:
        buffer.append(line)
        counter += 1
        if len(buffer) % 1000000 == 0:
            write_buffer(buffer, f"part{int(counter/1000000)}.jsonl")
            buffer = []