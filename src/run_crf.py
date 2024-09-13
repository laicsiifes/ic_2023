import os
import numpy as np
import optuna

from src.ner.corpora import read_corpus_file
from src.ner.nlp_parser import data_preprocessing
from src.ner.machine_learning import convert_data
from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report
from src.ner.utils import dump_report


def objective(trial):
    params = {
        'c1': trial.suggest_float('c1', low=0, high=2, step=0.2),
        'c2': trial.suggest_float('c2', low=0, high=2, step=0.2),
        'max_iterations': trial.suggest_int('max_iterations', low=0, high=2000, step=100),
    }
    crf_ = CRF(**params, algorithm='lbfgs', all_possible_transitions=True)
    crf_.fit(X_train, y_train)
    y_pred_ = crf_.predict(X_val)
    dict_report_ = classification_report(y_val, y_pred_, output_dict=True)
    return dict_report_['micro avg']['f1-score']


if __name__ == '__main__':

    corpus_name = 'harem_selective'
    # corpus_name = 'harem_total'
    # corpus_name = 'le_ner'
    # corpus_name = 'editais'
    # corpus_name = 'c_corpus'
    # corpus_name = 'pl_corpus'
    # corpus_name = 'bete'

    delimiter = '\t'

    corpus_dir = '../data/corpora'
    report_dir = '../data/results/'

    train_file = None
    val_file = None
    test_file = None

    idx = 1

    if corpus_name == 'harem_total':
        train_file = 'harem/train_total.txt'
        val_file = 'harem/dev_total.txt'
        test_file = 'harem/test_total.txt'
        delimiter = ' '
        idx = 3
    elif corpus_name == 'harem_selective':
        train_file = 'harem/train_selective.txt'
        val_file = 'harem/dev_selective.txt'
        test_file = 'harem/test_selective.txt'
        delimiter = ' '
        idx = 3
    elif corpus_name == 'le_ner':
        train_file = 'le_ner/train_clean.conll'
        val_file = 'le_ner/dev_clean.conll'
        test_file = 'le_ner/test_clean.conll'
        delimiter = ' '
    elif corpus_name == 'editais':
        train_file = 'editais/train.conll'
        val_file = 'editais/validation.conll'
        test_file = 'editais/test.conll'
        delimiter = ' '
    elif corpus_name == 'c_corpus':
        train_file = 'ulysses-ner-br/annotated-corpora/C_corpus_conll/c_corpus_tipos/train.txt'
        val_file = 'ulysses-ner-br/annotated-corpora/C_corpus_conll/c_corpus_tipos/valid.txt'
        test_file = 'ulysses-ner-br/annotated-corpora/C_corpus_conll/c_corpus_tipos/test.txt'
        delimiter = ' '
        idx = 1
    elif corpus_name == 'pl_corpus':
        train_file = 'ulysses-ner-br/annotated-corpora/PL_corpus_conll/pl_corpus_tipos/train.txt'
        val_file = 'ulysses-ner-br/annotated-corpora/PL_corpus_conll/pl_corpus_tipos/valid.txt'
        test_file = 'ulysses-ner-br/annotated-corpora/PL_corpus_conll/pl_corpus_tipos/test.txt'
        delimiter = ' '
        idx = 1
    elif corpus_name == 'bete':
        train_file = 'bete/train.txt'
        val_file = 'bete/eval.txt'
        test_file = 'bete/test.txt'
        delimiter = ' '
    else:
        print('Corpus option invalid!')
        exit(0)

    train_file = os.path.join(corpus_dir, train_file)
    val_file = os.path.join(corpus_dir, val_file)
    test_file = os.path.join(corpus_dir, test_file)

    report_dir = os.path.join(report_dir, corpus_name)

    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, corpus_name + '_crf.csv')

    train_data = read_corpus_file(train_file, delimiter=delimiter, ner_column=idx)
    val_data = read_corpus_file(val_file, delimiter=delimiter, ner_column=idx)
    test_data = read_corpus_file(test_file, delimiter=delimiter, ner_column=idx)
    
    test_data_original = np.array(test_data, dtype=object)

    print(f'\nCorpus: {corpus_name}')

    print(f'\n  Train data: {len(train_data)}')
    print(f'  Validation data: {len(val_data)}')
    print(f'  Test data: {len(test_data)}')

    print('\nPreprocessing ...')

    print('\n  Train data')

    train_data = data_preprocessing(train_data)

    print('  Validation data')

    val_data = data_preprocessing(val_data)

    print('  Test data')

    test_data = data_preprocessing(test_data)

    X_train, y_train = convert_data(train_data)
    X_val, y_val = convert_data(val_data)
    X_test, y_test = convert_data(test_data)

    print(f'\nExample features: {X_train[0]}')
    print(f'Tags: {y_train[0]}')

    study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=1)

    print('\nBest trial: ')

    best_trial = study.best_trial

    print(f'\n\tValue: {best_trial.value:.3f}')

    print('\tParams: ')

    for key, value in best_trial.params.items():
        print(f'\t  {key}: {value}')

    crf = CRF(**best_trial.params, algorithm='lbfgs', all_possible_transitions=True)

    print('\nEvaluating CRF')

    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    dict_report = classification_report(y_test, y_pred, output_dict=True)

    data_conll = ''

    for data, real_tags, pred_tags in \
            zip(test_data, y_test, y_pred):
        words = data[0]
        sent = '\n'.join('{0} {1} {2}'.format(word, real_tag, pred_tag)
                         for word, real_tag, pred_tag in
                         zip(words, real_tags, pred_tags))
        sent += '\n\n'
        data_conll += sent

    print(f'\nReport: {dict_report}')

    print(f'\nSaving the report in: {report_file}')

    dump_report(dict_report, report_file)

    script_result_file = os.path.join(report_dir, f'{corpus_name}_crf.tsv')

    with open(script_result_file, 'w', encoding='utf-8') as file:
        file.write(data_conll)
