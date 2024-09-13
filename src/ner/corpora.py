def read_corpus_file(corpus_file, delimiter='\t', ner_column=1):
    with open(corpus_file, encoding='utf-8') as file:
        lines = file.readlines()
    data = []
    words = []
    tags = []
    for line in lines:
        line = line.replace('\n', '')
        if line != '':
            if delimiter in line:
                fragments = line.split(delimiter)
                words.append(fragments[0])
                tags.append(fragments[ner_column])
        else:
            if len(words) > 1:
                data.append((words, tags))
            words = []
            tags = []
    return data
