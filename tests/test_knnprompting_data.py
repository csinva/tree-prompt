import pytest
import tprompt.data


dataset_names = ['agnews', 'cb', 'cr', 'dbpedia', 'mpqa', 'mr', 'rte', 'sst2', 'subj', 'trec']
@pytest.mark.parametrize("dataset_name", dataset_names)
def test_dataset(dataset_name):
    X_train_text, X_test_text, y_train, y_test = (
        tprompt.data.load_knnprompting_dataset(
            dataset_name, -1
        )
    )
    assert len(X_train_text) > 0
    assert len(y_train) > 0
    assert len(X_test_text) > 0
    assert len(y_test) > 0


@pytest.mark.parametrize("dataset_name", dataset_names)
def test_dataset_subsample(dataset_name):
    X_train_text, X_test_text, y_train, y_test = (
        tprompt.data.load_knnprompting_dataset(
            dataset_name, 100
        )
    )
    assert len(X_train_text) == 100
    assert len(y_train) == 100

