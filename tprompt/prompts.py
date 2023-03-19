from transformers import AutoTokenizer
from tprompt.utils import load_lm
import argparse
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
# from joblib import Memory
import joblib
import os
from dict_hash import sha256
from os.path import dirname, basename, join
path_to_repo = dirname(dirname(os.path.abspath(__file__)))

def get_verbalizer(args):
    VERB_0 = {0: ' Negative.', 1: ' Positive.'}
    VERB_1 = {0: ' No.', 1: ' Yes.', }
    VERB_FFB_0 = {0: ' Negative.', 1: ' Neutral.', 2: ' Positive.'}
    VERB_FFB_1 = {0: ' No.', 1: ' Maybe.', 2: ' Yes.'}
    VERB_LIST_DEFAULT = [VERB_0, VERB_1]

    # keys are (dataset_name, binary_classification)
    DATA_OUTPUT_STRINGS = {
        ('rotten_tomatoes', 1): VERB_LIST_DEFAULT,
        ('sst2', 1): VERB_LIST_DEFAULT,
        ('imdb', 1): VERB_LIST_DEFAULT,
        ('emotion', 1): VERB_LIST_DEFAULT,
        ('financial_phrasebank', 1): VERB_LIST_DEFAULT,
        ('financial_phrasebank', 0): [VERB_FFB_0, VERB_FFB_1],
    }
     #.get(args.dataset_name, VERB_LIST_DEFAULT)[args.verbalizer_num]
    return DATA_OUTPUT_STRINGS[(args.dataset_name, args.binary_classification)][args.verbalizer_num]
    

PROMPTS_MOVIE_0 = list(set([
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
    ' I thought the soundtrack of the movie was',
    ' I thought the originality of the movie was',
    ' I thought the action of the movie was',
    ' I thought the pacing of the movie was',
    ' I thought the length of the movie was',

    # Chat-GPT-Generated
    # > Generate more prompts for classifying movie review sentiment as Positive or Negative given these examples:
    ' The pacing of the movie was',
    ' The soundtrack of the movie was',
    ' The production design of the movie was',
    ' The chemistry between the actors was',
    ' The emotional impact of the movie was',
    ' The ending of the movie was',
    ' The themes explored in the movie were',
    ' The costumes in the movie were',
    ' The use of color in the movie was',
    ' The cinematography of the movie captured',
    ' The makeup and hair in the movie were',
    ' The lighting in the movie was',
    ' The sound design in the movie was',
    ' The humor in the movie was',
    ' The drama in the movie was',
    ' The social commentary in the movie was',
    ' The chemistry between the leads was',
    ' The relevance of the movie to the current times was',
    ' The depth of the story in the movie was',
    ' The cinematography in the movie was',
    ' The sound design in the movie was',
    ' The special effects in the movie were',
    ' The characters in the movie were',
    ' The plot of the movie was',
    ' The script of the movie was',
    ' The directing of the movie was',
    ' The performances in the movie were',
    ' The editing of the movie was',
    ' The climax of the movie was',
    ' The suspense in the movie was',
    ' The emotional impact of the movie was',
    ' The message of the movie was',
    ' The use of humor in the movie was',
    ' The use of drama in the movie was',
    ' The soundtrack of the movie was',
    ' The visual effects in the movie were',
    ' The themes explored in the movie were',
    ' The portrayal of relationships in the movie was',
    ' The exploration of societal issues in the movie was',
    ' The way the movie handles its subject matter was',
    ' The way the movie handles its characters was',
    ' The way the movie handles its plot twists was',
    ' The way the movie handles its narrative structure was',
    ' The way the movie handles its tone was',
    ' The casting of the film was',
    ' The writing of the movie was',
    ' The character arcs in the movie were',
    ' The dialogue in the movie was',
    ' The performances in the movie were',
    ' The chemistry between the actors in the movie was',
    ' The cinematography in the movie was',
    ' The visual effects in the movie were',
    ' The soundtrack in the movie was',
    ' The editing in the movie was',
    ' The direction of the movie was',
    ' The use of color in the movie was',
    ' The costume design in the movie was',
    ' The makeup and hair in the movie were',
    ' The special effects in the movie were',
    ' The emotional impact of the movie was',
    ' The ending of the movie was',
    ' The overall message of the movie was',
    ' The genre of the movie was well-executed',
    ' The casting choices for the movie were well-suited',
    ' The humor in the movie was effective',
    ' The drama in the movie was compelling',
    ' The suspense in the movie was well-maintained',
    ' The horror elements in the movie were well-done',
    ' The romance in the movie was believable',
    ' The action scenes in the movie were intense',
    ' The storyline of the movie was engaging'

    # > Generate nuanced prompts for classifying movie review sentiment as Positive or Negative.
    ' The movie had some flaws, but overall it was',
    " Although the movie wasn't perfect, I still thought it was",
    " The movie had its ups and downs, but ultimately it was",
    " The movie was a mixed bag, with some parts being",
    ' I have mixed feelings about the movie, but on the whole I would say it was',
    " The movie had some redeeming qualities, but I couldn't help feeling",
    ' The movie was entertaining, but lacked depth',
    ' The movie had a powerful message, but was poorly executed',
    ' Despite its flaws, I found the movie to be',
    ' The movie was technically impressive, but emotionally unengaging',
    ' The movie was thought-provoking, but also frustrating',
    ' The movie had moments of brilliance, but was ultimately disappointing',
    ' Although the movie had some good performances, it was let down by',
    ' The movie had a strong start, but faltered in the second half',
    ' The movie was well-made, but ultimately forgettable',
    ' The movie was engaging, but also emotionally exhausting',
    ' The movie was challenging, but also rewarding',
    " Although it wasn't perfect, the movie was worth watching because of"
    ' The movie was a thrilling ride, but also a bit cliché',
    ' The movie was visually stunning, but lacked substance',
]))

PROMPTS_FINANCE_0 = sorted(list(set([
    ' The financial sentiment of this phrase is',
    ' The senement of this sentence is',
    ' The general tone here is',
    ' I feel the sentiment is',
    ' The feeling for the economy here was',
    " Based on this the company's outlook will be",
    ' Earnings were',
    ' Long term forecasts are',
    ' Short-term forecasts are',
    ' Profits are',
    ' Revenue was',
    ' Investments are',
    ' Financial signals are',
    ' All indicators look',

    # Chat-GPT-Generated
    # > Generate more prompts for classifying financial sentences as Positive or Negative given these examples:
    'Overall, the financial outlook seems to be',
    'In terms of financial performance, the company has been',
    'The financial health of the company appears to be',
    'The market reaction to the latest earnings report has been',
    "The company's financial statements indicate that",
    "Investors' sentiment towards the company's stock is",
    'The financial impact of the recent economic events has been',
    "The company's financial strategy seems to be",
    'The financial performance of the industry as a whole has been',
    'The financial situation of the company seems to be',

    # > Generate nuanced prompts for classifying finfancial sentences as Positive or Negative.
    'Overall, the assessement of the financial performance of the company is',
    "The company's earnings exceeded expectations:",
    "The company's revenue figures were",
    "The unexpected financial surprises were",
    "Investments are",
    "Profits were",
    'Financial setbacks were',
    'Investor expectations are',
    'Financial strategy was',

    # > Generate different prompts for classifying financial sentences, that end with "Positive" or "Negative".
    'Based on the latest financial report, the overall financial sentiment is likely to be',
    'The financial health of the company seems to be trending',
    "The company's earnings for the quarter were",
    "Investors' sentiment towards the company's stock appears to be",
    "The company's revenue figures are expected to be",
    "The company's financial performance is expected to have what impact on the market:",
    "The latest financial report suggests that the company's financial strategy has been",
])))

def get_prompts(args, X_train_text, y_train, verbalizer, seed=1):
    assert args.prompt_source in ['manual', 'data_demonstrations']
    # random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    if args.prompt_source == 'manual':
        if args.dataset_name in ['rotten_tomatoes', 'sst2', 'imdb']:
            return PROMPTS_MOVIE_0
        elif args.dataset_name in ['financial_phrasebank']:
            return PROMPTS_FINANCE_0
    elif args.prompt_source == 'data_demonstrations':
        template = args.template_data_demonstrations
        # 1, 0 since positive usually comes first
        unique_ys = sorted(list(set(y_train)), key=lambda x: -x)
        examples_by_y = {}
        for y in unique_ys:
            examples_by_y[y] = sorted(
                list(filter(lambda ex: ex[1] == y, zip(X_train_text, y_train))))
        prompts = []
        while len(prompts) < args.num_prompts:
            prompt = ''
            for y in unique_ys:
                example = rng.choice(examples_by_y[y])
                text, _ = example
                prompt += template % (text, verbalizer[y]) + '\n'
            if prompt not in prompts:
                prompts.append(prompt)
        return prompts


def _calc_features_single_prompt(
    X_train_text, X_test_text, y_train, y_test, m, p
):
    """Calculate features with a single prompt (results get cached)
    """
    acc_baseline = max(y_train.mean(), 1 - y_train.mean())
    preds_train = m.predict(X_train_text)
    acc_train = np.mean(preds_train == y_train)

    preds_test = m.predict(X_test_text)
    acc_test = np.mean(preds_test == y_test)
    logging.info(f'prompt={p[:100]}... train:{acc_train:0.3f} baseline:{acc_baseline:0.3f} test:{acc_test:0.3f}')

    return preds_train, preds_test, acc_train

def engineer_prompt_features(
    args, X_train_text, X_test_text, y_train, y_test,
    # cache_dir=join(path_to_repo, 'results', 'cache_prompt_features'),
):
    logging.info('calculating prompt features with ' + args.checkpoint)
    args.prompt = 'Placeholder'

    # uses args.verbalizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = load_lm(checkpoint=args.checkpoint, tokenizer=tokenizer).to('cuda')
    m = tprompt.stump.PromptStump(
        args=args,
        split_strategy='manual',  # 'manual' specifies that we use args.prompt
        model=model,
        checkpoint=args.checkpoint,
    )

    # test different manual stumps
    prompts = get_prompts(args, X_train_text, y_train, m._get_verbalizer(), seed=1) # note, not passing seed here!
    prompt_features_train = np.zeros((len(X_train_text), len(prompts)))
    prompt_features_test = np.zeros((len(X_test_text), len(prompts)))
    accs_train = np.zeros(len(prompts))

    # compute features for prompts
    os.makedirs(args.cache_prompt_features_dir, exist_ok=True)
    for i, p in enumerate(tqdm(prompts)):
        m.prompt = p
        arg_names_cache=[
            'dataset_name',
            'binary_classification',
            'checkpoint',
            'verbalizer_num',
            'prompt_source',
            'template_data_demonstrations',
        ]
        args_dict_cache = {
            k: v for k, v in args._get_kwargs() if k in arg_names_cache
        }

        args_dict_cache['prompt'] = p
        save_dir_unique_hash = sha256(args_dict_cache)
        cache_file = join(args.cache_prompt_features_dir, f'{save_dir_unique_hash}.pkl')

        loaded_from_cache = False
        if os.path.exists(cache_file):
            # print('loading from cache!')
            try:
                preds_train, preds_test, acc_train = joblib.load(cache_file)
                loaded_from_cache = True
            except:
                pass
        
        if not loaded_from_cache:
            preds_train, preds_test, acc_train = _calc_features_single_prompt(
                X_train_text, X_test_text, y_train, y_test, m, p
            )
            joblib.dump((preds_train, preds_test, acc_train), cache_file)

        prompt_features_train[:, i] = preds_train
        prompt_features_test[:, i] = preds_test
        accs_train[i] = acc_train

    a = np.argsort(accs_train.flatten())[::-1]
    return prompt_features_train[:, a], prompt_features_test[:, a], np.array(prompts)[a].tolist()