import numpy as np
import tprompt.stump
import tprompt.tree
import tprompt.data
import random
from tqdm import tqdm
import imodelsx.data
import logging
import sklearn.tree
from transformers import AutoModelForCausalLM

def engineer_prompt_features(args, X_train_text, X_test_text, y_train, y_test):
    logging.info('calculating prompt features with ' + args.checkpoint)
    args.prompt = 'Placeholder'
        
    # uses args.verbalizer
    m = tprompt.stump.PromptStump(
        args=args,
        split_strategy='manual', # 'manual' specifies that we use args.prompt
        checkpoint=args.checkpoint,
    )

    # test different manual stumps
    prompts = [
        # ' What is the sentiment expressed by the reviewer for the movie?',
        # ' Is the movie positive or negative?',
        ' The movie is',
        ' Positive or Negative? The movie was',
        ' The sentiment of the movie was',
        ' The plot of the movie was really',
        ' The acting in the movie was',
        ' I felt the scenery was',
        ' The climax of the movie was',
        ' Overall I felt the acting was',
        ' I thought the visuals were generally',
    ]
    prompt_features_train = np.zeros((len(X_train_text), len(prompts)))
    prompt_features_test = np.zeros((len(X_test_text), len(prompts)))
    for i, p in enumerate(tqdm(prompts)):
        m.prompt = p
        preds_train = m.predict(X_train_text)
        prompt_features_train[:, i] = preds_train
        acc_baseline = max(y_train.mean(), 1 - y_train.mean())
        acc_train = np.mean(preds_train == y_train)
        print('prompt', p)
        # assert acc > acc_baseline, f'stump must improve acc but {acc:0.3f} <= {acc_baseline:0.2f}')
        print(f'\tacc_train {acc_train:0.3f} baseline: {acc_baseline:0.3f}')
        preds_test = m.predict(X_test_text)
        prompt_features_test[:, i] = preds_test
        acc_test = np.mean(preds_test == y_test)
        print(f'\tacc_test {acc_test:0.3f}')
    return prompt_features_train, prompt_features_test, prompts