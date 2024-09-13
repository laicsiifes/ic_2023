import os
import numpy as np

from src.bert_utils import read_corpus_file, extract_labels, replace_labels, tokenize_and_align_labels
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification,
                          EarlyStoppingCallback)
from transformers import TrainingArguments, Trainer
from seqeval.metrics import classification_report
from src.ner.utils import dump_report


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions_ = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[k] for k in label if k != -100] for label in labels]
    predicted_labels = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_, labels)
    ]
    all_metrics_ = classification_report(true_labels, predicted_labels, output_dict=True)
    metrics = {
        'precision': all_metrics_['micro avg']['precision'],
        'recall': all_metrics_['micro avg']['recall'],
        'f1': all_metrics_['micro avg']['f1-score']
    }
    return metrics


if __name__ == '__main__':

    # corpus_name = 'harem_selective'
    # corpus_name = 'harem_total'
    # corpus_name = 'le_ner'
    # corpus_name = 'editais'
    # corpus_name = 'c_corpus'
    # corpus_name = 'pl_corpus'
    corpus_name = 'bete'

    model_name = 'distilbertimbau'
    # model_name = 'bertimbau_base'
    # model_name = 'legal_bert_pt'
    # model_name = 'bio_bert'

    num_epochs = 100

    batch_size = 16

    corpus_dir = '../data/corpora'
    model_dir = '../data/models/bert'
    report_dir = '../data/results/'

    train_file = None
    val_file = None
    test_file = None

    idx = 1

    if corpus_name == 'le_ner':
        train_file = 'le_ner/train_clean.conll'
        val_file = 'le_ner/dev_clean.conll'
        test_file = 'le_ner/test_clean.conll'
        delimiter = ' '
    elif corpus_name == 'harem_total':
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
    elif corpus_name == 'bete':
        train_file = 'bete/train.txt'
        val_file = 'bete/eval.txt'
        test_file = 'bete/test.txt'
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
    elif corpus_name == 'editais':
        train_file = 'editais/train.conll'
        val_file = 'editais/validation.conll'
        test_file = 'editais/test.conll'
        delimiter = ' '
        idx = 1
    else:
        print('Corpus option invalid!')
        exit(0)

    print(f'\nCorpus: {corpus_name}')

    if model_name == 'distilbertimbau':
        model_checkpoint = 'adalbertojunior/distilbert-portuguese-cased'
    elif model_name == 'bertimbau_base':
        model_checkpoint = 'neuralmind/bert-base-portuguese-cased'
    elif model_name == 'legal_bert_pt':
        model_checkpoint = 'raquelsilveira/legalbertpt_sc'
    elif model_name == 'bio_bert':
        model_checkpoint = 'pucpr/biobertpt-bio'
    else:
        print('Embedding Model Name Option Invalid!')
        exit(0)

    report_dir = os.path.join(report_dir, corpus_name)

    training_model_dir = os.path.join(model_dir, corpus_name, model_name, 'training')
    best_model_dir = os.path.join(model_dir, corpus_name, model_name, 'best_model')

    os.makedirs(training_model_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    train_file = os.path.join(corpus_dir, train_file)
    val_file = os.path.join(corpus_dir, val_file)
    test_file = os.path.join(corpus_dir, test_file)

    train_data = read_corpus_file(train_file, delimiter=delimiter, ner_column=idx)
    val_data = read_corpus_file(val_file, delimiter=delimiter, ner_column=idx)
    test_data = read_corpus_file(test_file, delimiter=delimiter, ner_column=idx)

    label_names = extract_labels(labels=train_data['labels'])

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in id2label.items()}

    train_data = replace_labels(train_data, label2id)
    test_data = replace_labels(test_data, label2id)
    val_data = replace_labels(val_data, label2id)

    train_ds = Dataset.from_dict({
        'tokens': train_data['tokens'],
        'ner_tags': train_data['labels']
    })

    test_ds = Dataset.from_dict({
        'tokens': test_data['tokens'],
        'ner_tags': test_data['labels']
    })

    val_ds = Dataset.from_dict({
        'tokens': val_data['tokens'],
        'ner_tags': val_data['labels']
    })

    raw_datasets = DatasetDict({
        'train': train_ds,
        'test': test_ds,
        'validation': val_ds
    })

    print(f'\nTrain: {len(train_ds)}')
    print(f'Validation: {len(val_ds)}')
    print(f'Test: {len(test_ds)}\n')

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=batch_size,
        remove_columns=raw_datasets['train'].column_names,
        fn_kwargs={
            'tokenizer': tokenizer
        }
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id
    )

    logging_eval_steps = len(tokenized_datasets['train']) // batch_size

    args = TrainingArguments(
        output_dir=training_model_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=num_epochs,
        eval_steps=logging_eval_steps,
        logging_steps=logging_eval_steps,
        save_total_limit=1,
        weight_decay=0.01,
        push_to_hub=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    if os.path.exists(training_model_dir) and len(os.listdir(training_model_dir)) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(best_model_dir)

    predictions = trainer.predict(test_dataset=tokenized_datasets['test'])

    predictions = np.argmax(predictions.predictions, axis=-1)

    test_labels = tokenized_datasets['test']['labels']

    true_test_labels = [[label_names[k] for k in label if k != -100] for label in test_labels]

    predicted_test_labels = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, test_labels)
    ]

    dict_report = classification_report(true_test_labels, predicted_test_labels, output_dict=True)

    print('\n\n', dict_report)

    report_file = os.path.join(report_dir, f'full_{model_name}.csv')

    dump_report(dict_report, report_file)
