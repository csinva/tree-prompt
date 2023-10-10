<h1 align="center"> Tree Prompting </h1>
<p align="center"> Tree Prompting: Efficient Task Adaptation without Fine-Tuning, code for the <a href="">Tree-prompt paper</a>. 
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6+-blue">
  <img src="https://img.shields.io/pypi/v/imodelsx?color=green">  
</p>  

<p align="center"> Tree Prompting uses training examples to learn a tree of prompts to make a classificationg, yielding higher accuracy and better efficiency that baseline ensembles.
</p>

### Quickstart

For a simple scikit-learn interface to use Tree-Prompt, use the <a href="https://github.com/csinva/imodelsX">imodelsX package</a>. Installation: `pip install imodelsx`

```python
from imodelsx import TreePromptClassifier
import datasets
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# set up data
rng = np.random.default_rng(seed=42)
dset_train = datasets.load_dataset('rotten_tomatoes')['train']
dset_train = dset_train.select(rng.choice(
    len(dset_train), size=100, replace=False))
dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
dset_val = dset_val.select(rng.choice(
    len(dset_val), size=100, replace=False))

# set up arguments
prompts = [
    "This movie is",
    " Positive or Negative? The movie was",
    " The sentiment of the movie was",
    " The plot of the movie was really",
    " The acting in the movie was",
]
verbalizer = {0: " Negative.", 1: " Positive."}
checkpoint = "gpt2"

# fit model
m = TreePromptClassifier(
    checkpoint=checkpoint,
    prompts=prompts,
    verbalizer=verbalizer,
    cache_prompt_features_dir=None,  # 'cache_prompt_features_dir/gp2',
)
m.fit(dset_train["text"], dset_train["label"])


# compute accuracy
preds = m.predict(dset_val['text'])
print('\nTree-Prompt acc (val) ->',
      np.mean(preds == dset_val['label']))  # -> 0.7

# compare to accuracy for individual prompts
for i, prompt in enumerate(prompts):
    print(i, prompt, '->', m.prompt_accs_[i])  # -> 0.65, 0.5, 0.5, 0.56, 0.51

# visualize decision tree
plot_tree(
    m.clf_,
    fontsize=10,
    feature_names=m.feature_names_,
    class_names=list(verbalizer.values()),
    filled=True,
)
plt.show()
```

Reference:
```r
@misc{ch2022augmenting,
    title={Tree Prompting: Efficient Task Adaptation without Fine-Tuning},
    year={2023},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```


# Reproducing experiments

### Organization
- `tprompt`: contains main code for modeling (e.g. model architecture)
- `experiments`: code for runnning experiments (e.g. loading data, training models, evaluating models)
- `scripts`: scripts for running experiments (e.g. python scripts that launch jobs in `experiments` folder with different hyperparams)
- `notebooks`: jupyter notebooks for analyzing results and making figures
- `tests`: unit tests

### Setup
- clone and run `pip install -e .`, resulting in a package named `tprompt` that can be imported
    - see `setup.py` for dependencies, not all are required
- example run: run `python scripts/01_train_basic_models.py` (which calls `experiments/01_train_model.py` then view the results in `notebooks/01_model_results.ipynb`
