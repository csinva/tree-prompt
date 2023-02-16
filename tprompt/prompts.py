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

PROMPTS_MOVIE_0 = [
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
        ' How does the viewer feel about the movie?',
        ' What sentiment does the writer express for the movie?',
        ' Did the reviewer enjoy the movie?',
        ' The cinematography of the film was',
        ' The casting of the film was',
        ' I thought the soundtrack of the movie was',
        ' I thought the originality of the movie was',
        ' I thought the action of the movie was',
        ' I thought the pacing of the movie was',
        ' I thought the length of the movie was',
]

def engineer_prompt_features(args, X_train_text, X_test_text, X_cv_text, y_train, y_test, y_cv):
    logging.info('calculating prompt features with ' + args.checkpoint)
    args.prompt = 'Placeholder'
        
    # uses args.verbalizer
    m = tprompt.stump.PromptStump(
        args=args,
        split_strategy='manual', # 'manual' specifies that we use args.prompt
        checkpoint=args.checkpoint,
    )

    # test different manual stumps
    prompts = PROMPTS_MOVIE_0
    prompt_features_train = np.zeros((len(X_train_text), len(prompts)))
    prompt_features_test = np.zeros((len(X_test_text), len(prompts)))
    prompt_features_cv = np.zeros((len(X_cv_text), len(prompts)))
    accs_cv = np.zeros(len(prompts))
    for i, p in enumerate(tqdm(prompts)):
        m.prompt = p
        
        acc_baseline = max(y_train.mean(), 1 - y_train.mean())
        logging.info('prompt ' + p)
        preds_train = m.predict(X_train_text)
        acc_train = np.mean(preds_train == y_train)
        prompt_features_train[:, i] = preds_train
        logging.info(f'\tacc_train {acc_train:0.3f} baseline: {acc_baseline:0.3f}')

        preds_test = m.predict(X_test_text)
        acc_test = np.mean(preds_test == y_test)
        prompt_features_test[:, i] = preds_test
        logging.info(f'\tacc_test {acc_test:0.3f}')

        preds_cv = m.predict(X_cv_text)
        acc_cv = np.mean(preds_cv == y_cv)
        prompt_features_cv[:, i] = preds_cv
        logging.info(f'\tacc_cv {acc_cv:0.3f}')
        accs_cv[i] = acc_cv

    a = np.argsort(accs_cv.flatten())[::-1]
    return prompt_features_train[:, a], prompt_features_test[:, a], prompt_features_cv[:, a], np.array(prompts)[:, a].tolist()