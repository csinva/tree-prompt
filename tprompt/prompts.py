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
from joblib import Memory
import os
from os.path import dirname, basename, join
path_to_repo = dirname(dirname(os.path.abspath(__file__)))

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


def get_prompts(args, X_train_text, y_train, verbalizer, seed=1234):
    assert args.prompt_source in ['manual', 'data_demonstrations']
    random.seed(seed)
    if args.prompt_source == 'manual':
        return PROMPTS_MOVIE_0
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
                example = random.choice(examples_by_y[y])
                text, _ = example
                prompt += template % (text, verbalizer[y]) + '\n'
            if prompt not in prompts:
                prompts.append(prompt)
        return prompts


def engineer_prompt_features(
    args, X_train_text, X_test_text, y_train, y_test,
    cache_dir=join(path_to_repo, 'results', 'cache_features'),
    arg_names_cache=['checkpoint', 'verbalizer_num',
                     'prompt_source', 'template_data_demonstrations'],
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
    prompts = get_prompts(args, X_train_text, y_train, m._get_verbalizer())
    prompt_features_train = np.zeros((len(X_train_text), len(prompts)))
    prompt_features_test = np.zeros((len(X_test_text), len(prompts)))
    accs_train = np.zeros(len(prompts))

    # compute features for prompts
    for i, p in enumerate(tqdm(prompts)):
        m.prompt = p

        def _calc_features_single_prompt(
            args, X_train_text, X_test_text, y_train, y_test, m, p
        ):
            logging.info('prompt ' + p)
            acc_baseline = max(y_train.mean(), 1 - y_train.mean())
            preds_train = m.predict(X_train_text)
            acc_train = np.mean(preds_train == y_train)
            logging.info(
                f'\tacc_train {acc_train:0.3f} baseline: {acc_baseline:0.3f}')

            preds_test = m.predict(X_test_text)
            acc_test = np.mean(preds_test == y_test)
            logging.info(f'\tacc_test {acc_test:0.3f}')

            return preds_train, preds_test, acc_train

        compute_func = _calc_features_single_prompt
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            memory = Memory(cache_dir, verbose=0)
            compute_func = memory.cache(compute_func)

        args_cache = argparse.Namespace(
            **{k: v for k, v in args._get_kwargs() if k in arg_names_cache}
        )
        preds_train, preds_test, acc_train = \
            compute_func(args_cache, X_train_text,
                         X_test_text, y_train, y_test, m, p)

        prompt_features_train[:, i] = preds_train
        prompt_features_test[:, i] = preds_test
        accs_train[i] = acc_train

    a = np.argsort(accs_train.flatten())[::-1]
    return prompt_features_train[:, a], prompt_features_test[:, a], np.array(prompts)[a].tolist()
