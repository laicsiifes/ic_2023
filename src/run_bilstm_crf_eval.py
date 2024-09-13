import os

from seqeval.metrics import classification_report
from src.bert_utils import dump_report


if __name__ == '__main__':

    # corpus_name = 'harem_selective'
    # corpus_name = 'harem_total'
    # corpus_name = 'le_ner'
    # corpus_name = 'editais'
    corpus_name = 'c_corpus'
    # corpus_name = 'pl_corpus'
    # corpus_name = 'bete'


    model_dir = f'../data/models/bilstm/{corpus_name}'

    report_dir = f'../data/results/{corpus_name}'

    os.makedirs(report_dir, exist_ok=True)

    list_model_names = os.listdir(model_dir)

    columns_names = ['tokens', 'real_tag', 'predicted_tag']

    for model_name in list_model_names:

        print(f'\nModel: {model_name}')

        tsv_file = os.path.join(model_dir, model_name, 'test.tsv')

        if not os.path.exists(tsv_file):
            continue

        with open(file=tsv_file, mode='r') as file:
            lines = file.readlines()

        all_y_test = []
        all_y_pred = []

        y_test = []
        y_pred = []

        for line in lines:

            line = line.replace('\n', '').strip()

            fragments = line.split(' ')

            if len(fragments) == 3:
                y_test.append(fragments[1])
                y_pred.append(fragments[2])
            else:
                all_y_test.append(y_test.copy())
                all_y_pred.append(y_pred.copy())
                y_test.clear()
                y_pred.clear()

        dict_report = classification_report(all_y_test, all_y_pred, output_dict=True)

        results_file_path = os.path.join(report_dir, f'bilstm_crf_{model_name}.csv')

        dump_report(dict_report, results_file_path)
