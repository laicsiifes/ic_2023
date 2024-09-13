import os

from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import SGD


if __name__ == '__main__':

    corpus_name = 'harem_selective'
    # corpus_name = 'harem_total'
    # corpus_name = 'le_ner'
    # corpus_name = 'editais'
    # corpus_name = 'c_corpus'
    # corpus_name = 'pl_corpus'
    # corpus_name = 'bete'

    embedding_model_name = 'distilbertimbau'
    # embedding_model_name = 'bertimbau_base'
    # embedding_model_name = 'legal_bert_pt'
    # embedding_model_name = 'bio_bert'

    is_use_crf = True

    n_epochs = 100

    batch_size = 32

    delimiter = '\t'

    model_dir = '../data/models/bilstm'

    corpus_dir = None

    train_file = None
    val_file = None
    test_file = None

    idx = 1

    if corpus_name == 'le_ner':
        columns_dict = {
            0: 'token',
            1: 'label'
        }
        corpus_dir = '../data/corpora/le_ner/'
        train_file = 'train_clean.conll'
        val_file = 'dev_clean.conll'
        test_file = 'test_clean.conll'
        delimiter = ' '
    elif corpus_name == 'harem_total':
        columns_dict = {
            0: 'token',
            1: 'pos',
            2: 'sublabel',
            3: 'label'
        }
        corpus_dir = '../data/corpora/harem/'
        train_file = 'train_total.txt'
        val_file = 'dev_total.txt'
        test_file = 'test_total.txt'
        delimiter = ' '
        idx = 3
    elif corpus_name == 'harem_selective':
        columns_dict = {
            0: 'token',
            1: 'pos',
            2: 'sublabel',
            3: 'label'
        }
        corpus_dir = '../data/corpora/harem/'
        train_file = 'train_selective.txt'
        val_file = 'dev_selective.txt'
        test_file = 'test_selective.txt'
        delimiter = ' '
        idx = 3
    elif corpus_name == 'bete':
        columns_dict = {
            0: 'token',
            1: 'label'
        }
        corpus_dir = '../data/corpora/bete/'
        train_file = 'train.txt'
        val_file = 'eval.txt'
        test_file = 'test.txt'
        delimiter = ' '
    elif corpus_name == 'c_corpus':
        columns_dict = {
            0: 'token',
            1: 'label'
        }
        corpus_dir = '../data/corpora/ulysses-ner-br/annotated-corpora/C_corpus_conll/c_corpus_tipos/'
        train_file = 'train.txt'
        val_file = 'valid.txt'
        test_file = 'test.txt'
        delimiter = ' '
        idx = 1
    elif corpus_name == 'pl_corpus':
        columns_dict = {
            0: 'token',
            1: 'label'
        }
        corpus_dir = '../data/corpora/ulysses-ner-br/annotated-corpora/PL_corpus_conll/pl_corpus_tipos/'
        train_file = 'train.txt'
        val_file = 'valid.txt'
        test_file = 'test.txt'
        delimiter = ' '
        idx = 1
    elif corpus_name == 'editais':
        columns_dict = {
            0: 'token',
            1: 'label'
        }
        corpus_dir = '../data/corpora/editais'
        train_file = 'train.conll'
        val_file = 'validation.conll'
        test_file = 'test.conll'
        delimiter = ' '
    else:
        print('Corpus Option Invalid!')
        exit(0)

    if embedding_model_name == 'distilbertimbau':
        embedding_model_path = 'adalbertojunior/distilbert-portuguese-cased'
    elif embedding_model_name == 'bertimbau_base':
        embedding_model_path = 'neuralmind/bert-base-portuguese-cased'
    elif embedding_model_name == 'legal_bert_pt':
        embedding_model_path = 'raquelsilveira/legalbertpt_sc'
    elif embedding_model_name == 'bio_bert':
        embedding_model_path = 'pucpr/biobertpt-bio'
    else:
        print('Embedding Model Name Option Invalid!')
        exit(0)

    model_dir = os.path.join(model_dir, corpus_name, embedding_model_name)

    os.makedirs(model_dir, exist_ok=True)

    print(f'\nCorpus: {corpus_name}')
    print(f'\nEmbedding Model Name: {embedding_model_name}\n')

    corpus = ColumnCorpus(corpus_dir, columns_dict, train_file=train_file, test_file=test_file, dev_file=val_file)

    print(f'\nTrain len: {len(corpus.train)}')
    print(f'Dev len: {len(corpus.dev)}')
    print(f'Test len: {len(corpus.test)}')

    print(f"\nTrain: {corpus.train[0].to_tagged_string('label')}")
    print(f"Dev: {corpus.dev[0].to_tagged_string('label')}")
    print(f"Test: {corpus.test[0].to_tagged_string('label')}")

    tag_type = 'label'

    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    print(f'\nTags: {tag_dictionary.idx2item}')

    bert_embedding = TransformerWordEmbeddings(model=embedding_model_path, layers='-1', subtoken_pooling='first',
                                               fine_tune=False, use_context=True, allow_long_sentences=True)

    tagger = SequenceTagger(hidden_size=256, embeddings=bert_embedding, tag_dictionary=tag_dictionary,
                            tag_type=tag_type, use_crf=is_use_crf)

    trainer = ModelTrainer(tagger, corpus)

    trainer.train(base_path=model_dir, optimizer=SGD, learning_rate=0.1, patience=5, mini_batch_size=batch_size,
                  max_epochs=n_epochs)
