from typing import Dict, Iterable, Tuple

import json
import os

import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import imodelsx.util

def load_huggingface_dataset(dataset_name, subsample_frac=1.0):
    """Load text dataset from huggingface (with train/validation spltis) + return the relevant dataset key
    """
    # load dset
    if dataset_name == 'tweet_eval':
        dset = datasets.load_dataset('tweet_eval', 'hate')
    elif dataset_name == 'financial_phrasebank':
        train = datasets.load_dataset('financial_phrasebank', 'sentences_75agree',
                                      revision='main', split='train')
        idxs_train, idxs_val = train_test_split(
            np.arange(len(train)), test_size=0.33, random_state=13)
        dset = datasets.DatasetDict()
        dset['train'] = train.select(idxs_train)
        dset['validation'] = train.select(idxs_val)
    else:
        dset = datasets.load_dataset(dataset_name)

    # process dset
    dataset_key_text = 'text'
    if dataset_name == 'sst2':
        dataset_key_text = 'sentence'
    elif dataset_name == 'financial_phrasebank':
        dataset_key_text = 'sentence'
    elif dataset_name == 'imdb':
        del dset['unsupervised']
        dset['validation'] = dset['test']

    # subsample datak
    if subsample_frac > 0:
        n = len(dset['train'])
        dset['train'] = dset['train'].select(np.random.choice(
            range(n), replace=False,
            size=int(n * subsample_frac)
        ))
    return dset, dataset_key_text

def convert_text_data_to_counts_array(
    X_train, X_test, ngrams=2, all_ngrams=True,
    tokenizer=None,
    ):
    if tokenizer == None:
        tokenizer = imodelsx.util.get_spacy_tokenizer()
        
    if all_ngrams:
        ngram_range=(1, ngrams)
    else:
        ngram_range=(ngrams, ngrams)

    v = CountVectorizer(
        ngram_range=ngram_range,
        tokenizer=tokenizer,
        lowercase=True,
        token_pattern=None,
    )
    X_train = v.fit_transform(X_train)
    X_test = v.transform(X_test)
    feature_names = v.get_feature_names_out().tolist()
    return X_train, X_test, feature_names


# https://github.com/BenfengXu/KNNPrompting/blob/050d7e455113c0afa82de1537210007c34e96e57/utils/dataset.py#L106
KNNPROMPTING_DATA_LABELS = {
    'agnews': {'1': 0, '2': 1, '3': 2, '4': 3},
    'cb': {'contradiction': 0, 'entailment': 1, 'neutral': 2},
    'cr': {'0': 0, '1': 1},
    'dbpedia': {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                         '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
                         '11': 10, '12': 11, '13': 12, '14': 13},
    'mpqa': {'0': 0, '1': 1},
    'mr': {'0': 0, '1': 1},
    'rte': {'not_entailment': 0, 'entailment': 1},
    'sst2': {'0': 0, '1': 1},
    'sst5': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
    'subj': {'0': 0, '1': 1},
    'trec': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
}

# https://github.com/BenfengXu/KNNPrompting/blob/050d7e455113c0afa82de1537210007c34e96e57/utils/template.py#L96
KNNPROMPTING_DATA_TEMPLATE_FNS = { 
    'agnews': lambda ins, label: f"input: {ins['sentence']}\ntype:",
    'cb': lambda ins, label: f"premise: {ins['premise']}\nhypothesis: {ins['hypothesis']}\nprediction:",
    'cr': lambda ins, label: f"Review: {ins['sentence']}\nSentiment:",
    'dbpedia': lambda ins, label: f"input: {ins['sentence']}\ntype:",
    'mpqa': lambda ins, label: f"Review: {ins['sentence']}\nSentiment:",
    'mr': lambda ins, label: f"Review: {ins['sentence']}\nSentiment:",
    'rte': lambda ins, label: f"premise: {ins['sentence_1']}\nhypothesis: {ins['sentence_2']}\nprediction:",
    'sst2': lambda ins, label: f"Question: {ins['sentence']}\nType:",
    'sst5': lambda ins, label: f"Review: {ins['sentence']}\nSentiment:",
    'subj': lambda ins, label: f"Input: {ins['sentence']}\nType:",
    'trec': lambda ins, label: f"Question: {ins['sentence']}\nType:",
}

KNNPROMPTING_VERBALIZERS = {
    'agnews': {'1': 'world', '2': 'sports', '3': 'business', '4': 'technology'},
    'cb': {'contradiction': 'false', 'entailment': 'true', 'neutral': 'neither'},
    'cr': {'0': 'negative', '1': 'positive'},
    'dbpedia': {'1': 'company', '2': 'school', '3': 'artist', '4': 'athlete', '5': 'politics',
                           '6': 'transportation', '7': 'building', '8': 'nature', '9': 'village', '10': 'animal',
                           '11': 'plant', '12': 'album', '13': 'film', '14': 'book'},
    'mpqa': {'0': 'negative', '1': 'positive'},
    'mr': {'0': 'negative', '1': 'positive'},
    'rte': {'not_entailment': 'false', 'entailment': 'true'},
    'sst2': {'0': 'negative', '1': 'positive'},
    'sst5':  {'0': 'terrible', '1': 'bad', '2': 'okay', '3': 'good', '4': 'great'},
    'subj': {'0': 'subjective', '1': 'objective'},
    'trec': {'0': 'description', '1': 'entity', '2': 'expression', '3': 'human','4': 'location', '5': 'number'},
}



def _load_knnprompting_dataset_file(
    data_file: str, dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mapping = KNNPROMPTING_DATA_LABELS[dataset_name]
    template_fn = KNNPROMPTING_DATA_TEMPLATE_FNS[dataset_name]
    text = []
    labels = []
    with open(data_file, 'r') as f:
        read_lines = f.readlines()
        for line in read_lines:
            instance = json.loads(line.strip())
            # assert instance['label'] in mapping, f"got invalid label {label} for mapping {mapping}"
            label = mapping[instance["label"]]
            labels.append(int(label))
            text.append(template_fn(instance, label))
    return np.array(text), np.array(labels)


def load_knnprompting_dataset(
    dataset_name: str, subsample_n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tprompt_dir_path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.pardir
        )
    )
    # load from tprompt/data/<dataset_name>
    dataset_folder = os.path.join(
        tprompt_dir_path, 'data', dataset_name
    )
    assert os.path.exists(dataset_folder), f"could not find dataset folder at path {dataset_folder}"
    # contains ['train.json', 'test.jsonl', 'dev_subsample.jsonl', 'classes.txt']

    assert dataset_name in KNNPROMPTING_DATA_LABELS, f'invalid dataset name {dataset_name}'
    label_mapping = KNNPROMPTING_DATA_LABELS[dataset_name]
    
    train_filename = 'train_subset.jsonl' if dataset_name == 'dbpedia' else 'train.jsonl'
    X_train, y_train = _load_knnprompting_dataset_file(
        os.path.join(dataset_folder, train_filename),
        dataset_name
    )

    if subsample_n > 0:
        if subsample_n < len(X_train):
            print(f"subsampling dataset {dataset_name} from length {len(X_train)} to {subsample_n}")
            idxs = np.random.choice(range(len(X_train)), size=subsample_n)
            X_train = X_train[idxs]
            y_train = y_train[idxs]
        else:
            print("dataset already small enough; not subsampling.")

    test_filename = 'dev_subsample.jsonl'
    # test_filename = 'test.jsonl'
    X_test, y_test = _load_knnprompting_dataset_file(
        os.path.join(dataset_folder, test_filename),
        dataset_name
    )
    return X_train, X_test, y_train, y_test


class KnnPromptVerbalizer:
    def __init__(self, dataset_name):
        self.id2label = {v: k for k,v in KNNPROMPTING_DATA_LABELS[dataset_name].items()}
        self.verbalizer = KNNPROMPTING_VERBALIZERS[dataset_name]
        self._values = [f" {v}" for v in self.verbalizer.values()]
    
    def __str__(self):
        return str(self.id2label)
    
    def __getitem__(self, id_num: int) -> str:
        label = self.id2label[id_num]
        return " " + self.verbalizer[label]
    
    def values(self) -> Iterable[int]:
        return self._values


def get_verbalizer_knnprompting(dataset_name: str):
    return KnnPromptVerbalizer(dataset_name)

