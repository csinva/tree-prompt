from typing import Dict, List
from transformers import AutoTokenizer
from tprompt.utils import load_lm, load_tokenizer
import numpy as np
import tprompt.stump
import tprompt.tree
import tprompt.data
from tqdm import trange, tqdm
import math
import logging
import joblib
import os
import random
from dict_hash import sha256
from os.path import dirname, basename, join

path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def get_verbalizer(args):
    if args.dataset_name.startswith("knnp__"):
        return tprompt.data.get_verbalizer_knnprompting(
            args.dataset_name.replace("knnp__", "")
        )

    VERB_0 = {0: " Negative.", 1: " Positive."}
    VERB_1 = {
        0: " No.",
        1: " Yes.",
    }
    VERB_FFB_0 = {0: " Negative.", 1: " Neutral.", 2: " Positive."}
    VERB_FFB_1 = {0: " No.", 1: " Maybe.", 2: " Yes."}
    # VERB_EMOTION_BINARY = {0: ' Sad.', 1: ' Happy.'}

    # note: verb=1 usually uses yes/no. We don't support this for emotion, since we must specify a value for each of 6 classes
    VERB_EMOTION_0 = {
        0: " Sad.",
        1: " Happy.",
        2: " Love.",
        3: " Anger.",
        4: " Fear.",
        5: " Surprise.",
    }
    # VERB_EMOTION_1 = {0: ' No.', 1: ' Maybe.', 2: ' Yes.'}

    VERB_LIST_DEFAULT = [VERB_0, VERB_1]

    # keys are (dataset_name, binary_classification)
    DATA_OUTPUT_STRINGS = {
        ("rotten_tomatoes", 1): VERB_LIST_DEFAULT,
        ("sst2", 1): VERB_LIST_DEFAULT,
        ("imdb", 1): VERB_LIST_DEFAULT,
        ("emotion", 1): VERB_LIST_DEFAULT,
        ("financial_phrasebank", 1): VERB_LIST_DEFAULT,
        ("financial_phrasebank", 0): [VERB_FFB_0, VERB_FFB_1],
        ("emotion", 0): [VERB_EMOTION_0],
    }
    # .get(args.dataset_name, VERB_LIST_DEFAULT)[args.verbalizer_num]
    return DATA_OUTPUT_STRINGS[(args.dataset_name, args.binary_classification)][
        args.verbalizer_num
    ]


PROMPTS_MOVIE_0 = list(
    set(
        [
            # ' What is the sentiment expressed by the reviewer for the movie?',
            # ' Is the movie positive or negative?',
            " The movie is",
            " Positive or Negative? The movie was",
            " The sentiment of the movie was",
            " The plot of the movie was really",
            " The acting in the movie was",
            " I felt the scenery was",
            " The climax of the movie was",
            " Overall I felt the acting was",
            " I thought the visuals were generally",
            " How does the viewer feel about the movie?",
            " What sentiment does the writer express for the movie?",
            " Did the reviewer enjoy the movie?",
            " The cinematography of the film was",
            " I thought the soundtrack of the movie was",
            " I thought the originality of the movie was",
            " I thought the action of the movie was",
            " I thought the pacing of the movie was",
            " I thought the length of the movie was",
            # Chat-GPT-Generated
            # > Generate more prompts for classifying movie review sentiment as Positive or Negative given these examples:
            " The pacing of the movie was",
            " The soundtrack of the movie was",
            " The production design of the movie was",
            " The chemistry between the actors was",
            " The emotional impact of the movie was",
            " The ending of the movie was",
            " The themes explored in the movie were",
            " The costumes in the movie were",
            " The use of color in the movie was",
            " The cinematography of the movie captured",
            " The makeup and hair in the movie were",
            " The lighting in the movie was",
            " The sound design in the movie was",
            " The humor in the movie was",
            " The drama in the movie was",
            " The social commentary in the movie was",
            " The chemistry between the leads was",
            " The relevance of the movie to the current times was",
            " The depth of the story in the movie was",
            " The cinematography in the movie was",
            " The sound design in the movie was",
            " The special effects in the movie were",
            " The characters in the movie were",
            " The plot of the movie was",
            " The script of the movie was",
            " The directing of the movie was",
            " The performances in the movie were",
            " The editing of the movie was",
            " The climax of the movie was",
            " The suspense in the movie was",
            " The emotional impact of the movie was",
            " The message of the movie was",
            " The use of humor in the movie was",
            " The use of drama in the movie was",
            " The soundtrack of the movie was",
            " The visual effects in the movie were",
            " The themes explored in the movie were",
            " The portrayal of relationships in the movie was",
            " The exploration of societal issues in the movie was",
            " The way the movie handles its subject matter was",
            " The way the movie handles its characters was",
            " The way the movie handles its plot twists was",
            " The way the movie handles its narrative structure was",
            " The way the movie handles its tone was",
            " The casting of the film was",
            " The writing of the movie was",
            " The character arcs in the movie were",
            " The dialogue in the movie was",
            " The performances in the movie were",
            " The chemistry between the actors in the movie was",
            " The cinematography in the movie was",
            " The visual effects in the movie were",
            " The soundtrack in the movie was",
            " The editing in the movie was",
            " The direction of the movie was",
            " The use of color in the movie was",
            " The costume design in the movie was",
            " The makeup and hair in the movie were",
            " The special effects in the movie were",
            " The emotional impact of the movie was",
            " The ending of the movie was",
            " The overall message of the movie was",
            " The genre of the movie was well-executed",
            " The casting choices for the movie were well-suited",
            " The humor in the movie was effective",
            " The drama in the movie was compelling",
            " The suspense in the movie was well-maintained",
            " The horror elements in the movie were well-done",
            " The romance in the movie was believable",
            " The action scenes in the movie were intense",
            " The storyline of the movie was engaging"
            # > Generate nuanced prompts for classifying movie review sentiment as Positive or Negative.
            " The movie had some flaws, but overall it was",
            " Although the movie wasn't perfect, I still thought it was",
            " The movie had its ups and downs, but ultimately it was",
            " The movie was a mixed bag, with some parts being",
            " I have mixed feelings about the movie, but on the whole I would say it was",
            " The movie had some redeeming qualities, but I couldn't help feeling",
            " The movie was entertaining, but lacked depth",
            " The movie had a powerful message, but was poorly executed",
            " Despite its flaws, I found the movie to be",
            " The movie was technically impressive, but emotionally unengaging",
            " The movie was thought-provoking, but also frustrating",
            " The movie had moments of brilliance, but was ultimately disappointing",
            " Although the movie had some good performances, it was let down by",
            " The movie had a strong start, but faltered in the second half",
            " The movie was well-made, but ultimately forgettable",
            " The movie was engaging, but also emotionally exhausting",
            " The movie was challenging, but also rewarding",
            " Although it wasn't perfect, the movie was worth watching because of"
            " The movie was a thrilling ride, but also a bit cliché",
            " The movie was visually stunning, but lacked substance",
        ]
    )
)

PROMPTS_FINANCE_0 = sorted(
    list(
        set(
            [
                " The financial sentiment of this phrase is",
                " The senement of this sentence is",
                " The general tone here is",
                " I feel the sentiment is",
                " The feeling for the economy here was",
                " Based on this the company's outlook will be",
                " Earnings were",
                " Long term forecasts are",
                " Short-term forecasts are",
                " Profits are",
                " Revenue was",
                " Investments are",
                " Financial signals are",
                " All indicators look",
                # Chat-GPT-Generated
                # > Generate more prompts for classifying financial sentences as Positive or Negative given these examples:
                "Overall, the financial outlook seems to be",
                "In terms of financial performance, the company has been",
                "The financial health of the company appears to be",
                "The market reaction to the latest earnings report has been",
                "The company's financial statements indicate that",
                "Investors' sentiment towards the company's stock is",
                "The financial impact of the recent economic events has been",
                "The company's financial strategy seems to be",
                "The financial performance of the industry as a whole has been",
                "The financial situation of the company seems to be",
                # > Generate nuanced prompts for classifying financial sentences as Positive or Negative.
                "Overall, the assessement of the financial performance of the company is",
                "The company's earnings exceeded expectations:",
                "The company's revenue figures were",
                "The unexpected financial surprises were",
                "Investments are",
                "Profits were",
                "Financial setbacks were",
                "Investor expectations are",
                "Financial strategy was",
                # > Generate different prompts for classifying financial sentences, that end with "Positive" or "Negative".
                "Based on the latest financial report, the overall financial sentiment is likely to be",
                "The financial health of the company seems to be trending",
                "The company's earnings for the quarter were",
                "Investors' sentiment towards the company's stock appears to be",
                "The company's revenue figures are expected to be",
                "The company's financial performance is expected to have what impact on the market:",
                "The latest financial report suggests that the company's financial strategy has been",
            ]
        )
    )
)

PROMPTS_EMOTION_0 = list(
    set(
        [
            " The emotion of this sentence is:",
            " This tweet contains the emotion",
            " The emotion of this tweet is",
            " I feel this tweet is related to ",
            " The feeling of this tweet was",
            " This tweet made me feel",
            # Chat-GPT-Generated
            # > Generate prompts for classifying tweets based on their emotion (e.g. joy, sadness, fear, etc.). The prompt should end with the emotion.
            " When I read this tweet, the emotion that came to mind was",
            " The sentiment expressed in this tweet is",
            " This tweet conveys a sense of",
            " The emotional tone of this tweet is",
            " This tweet reflects a feeling of",
            " The underlying emotion in this tweet is",
            " This tweet evokes a sense of",
            " The mood conveyed in this tweet is",
            " I perceive this tweet as being",
            " This tweet gives off a feeling of",
            " The vibe of this tweet is",
            " The atmosphere of this tweet suggests a feeling of",
            " The overall emotional content of this tweet is",
            " The affective state expressed in this tweet is",
            # > Generate language model prompts for classifying tweets based on their emotion (e.g. joy, sadness, fear, etc.). The prompt should end with the emotion.
            " Based on the content of this tweet, the emotion I would classify it as",
            " When reading this tweet, the predominant emotion that comes to mind is",
            " This tweet seems to convey a sense of",
            " I detect a feeling of",
            " If I had to categorize the emotion behind this tweet, I would say it is",
            " This tweet gives off a sense of",
            " When considering the tone and language used in this tweet, I would classify the emotion as",
            # > Generate unique prompts for detecting the emotion of a tweet (e.g. joy, sadness, surprise). The prompt should end with the emotion.
            # ' The emotion of this tweet is',
            " The main emotion in this sentence is",
            " The overall tone I sense is",
            " The mood I am in is",
            " Wow this made me feel",
            " This tweet expresses",
        ]
    )
)


def get_prompts(args, X_train_text, y_train, verbalizer, seed=1):
    assert args.prompt_source in ["manual", "data_demonstrations"]
    rng = np.random.default_rng(seed=seed)
    if args.prompt_source == "manual":
        if args.dataset_name in ["rotten_tomatoes", "sst2", "imdb"]:
            return PROMPTS_MOVIE_0
        elif args.dataset_name in ["financial_phrasebank"]:
            return PROMPTS_FINANCE_0
        elif args.dataset_name in ["emotion"]:
            return PROMPTS_EMOTION_0
        else:
            raise ValueError("need to set prompt in get_prompts!")
    elif args.prompt_source == "data_demonstrations":
        template = args.template_data_demonstrations
        # 1, 0 since positive usually comes first
        unique_ys = sorted(list(set(y_train)), key=lambda x: -x)
        examples_by_y = {}
        for y in unique_ys:
            examples_by_y[y] = sorted(
                list(filter(lambda ex: ex[1] == y, zip(X_train_text, y_train)))
            )
        prompts = []

        # Create num_prompts prompts
        prompt_pbar = tqdm(
            total=args.num_prompts, desc="building prompts", leave=False, colour="green"
        )
        while len(prompts) < args.num_prompts:
            # Create a prompt with demonstration for each class
            prompt = ""
            chosen_examples = {
                y: rng.choice(
                    examples,
                    size=args.num_data_demonstrations_per_class,
                    replace=(len(examples) < args.num_data_demonstrations_per_class),
                )
                for y, examples in examples_by_y.items()
            }

            # Take an even number of demonstrations per class, but shuffle them.
            demo_classes = unique_ys * math.ceil(args.num_data_demonstrations_per_class // len(unique_ys))
            random.shuffle(demo_classes)
            demo_classes = demo_classes[:args.num_data_demonstrations_per_class]

            for idx, y in enumerate(demo_classes):
                text, _ = chosen_examples[y][idx]
                prompt += template % (text, verbalizer[y]) + "\n"
            if prompt not in prompts:
                prompts.append(prompt)
                prompt_pbar.update(1)
        return prompts


def _calc_features_single_prompt(X, y, m, p, past_key_values=None):
    """Calculate features with a single prompt (results get cached)
    preds: np.ndarray[int] of shape (X.shape[0],)
        If multiclass, each int takes value 0, 1, ..., n_classes - 1 based on the verbalizer
    """
    m.prompt = p
    if past_key_values is not None:
        preds = m.predict_with_cache(X, past_key_values)
    else:
        preds = m.predict(X)
    acc = np.mean(preds == y)
    return preds, acc


def calc_prompt_features(
    args,
    prompts: List[str],
    X_train_text,
    X_test_text,
    y_train,
    y_test,
    checkpoint: str,
    verbalizer: Dict[int, str],
    cache_prompt_features_dir=join(path_to_repo, "results", "cache_prompt_features"),
):
    logging.info("calculating prompt features with " + checkpoint)
    tokenizer = load_tokenizer(checkpoint=checkpoint)
    model = load_lm(checkpoint=checkpoint, tokenizer=tokenizer).to("cuda")
    m = None  # don't load model until checking for cache to speed things up

    # test different manual stumps
    # print('prompts', prompts)
    prompt_features_train = np.zeros((len(X_train_text), len(prompts)))
    prompt_features_test = np.zeros((len(X_test_text), len(prompts)))
    accs_train = np.zeros(len(prompts))

    # compute features for prompts
    os.makedirs(cache_prompt_features_dir, exist_ok=True)
    for i, p in enumerate(tqdm(prompts)):
        # set up name of file for saving based on argument values
        arg_names_cache = [
            "dataset_name",
            "binary_classification",
            "checkpoint",
            "verbalizer_num",
            "prompt_source",
            "template_data_demonstrations",
        ]
        args_dict_cache = {k: v for k, v in args._get_kwargs() if k in arg_names_cache}
        args_dict_cache["prompt"] = p
        save_dir_unique_hash = sha256(args_dict_cache)
        cache_file = join(cache_prompt_features_dir, f"{save_dir_unique_hash}.pkl")

        # load from cache if possible
        loaded_from_cache = False
        if args.use_cache == 1 and os.path.exists(cache_file):
            print("loading from cache!")
            try:
                preds_train, preds_test, acc_train = joblib.load(cache_file)
                loaded_from_cache = True
            except:
                pass

        # actually compute prompt features (integer valued, 0, ..., n_classes - 1)
        if not loaded_from_cache:
            if m is None:
                m = tprompt.stump.PromptStump(
                    args=args,
                    split_strategy="manual",  # 'manual' specifies that we use m.prompt instead of autoprompting
                    model=model,
                    checkpoint=checkpoint,
                    verbalizer=verbalizer,
                    batch_size=args.batch_size,
                )

            # Compute max length between train and test
            # template = m.args.template_data_demonstrations
            # longest_verbalizer = sorted(m.verbalizer.values(), key=lambda v: len(m.tokenizer.encode(v)))
            # max_len_input = max(len(m.tokenizer.encode(template % (x, longest_verbalizer))) for x in X_train_text + X_test_text)

            # import pdb; pdb.set_trace()
            past_key_values = None
            if args.cache_prompt == 1:
                m.prompt = p
                past_key_values = m.calc_key_values(X_train_text)
            preds_train, acc_train = _calc_features_single_prompt(
                X_train_text, y_train, m, p, past_key_values=past_key_values
            )
            preds_test, _ = _calc_features_single_prompt(
                X_test_text, y_test, m, p, past_key_values=past_key_values
            )
            joblib.dump((preds_train, preds_test, acc_train), cache_file)

        prompt_features_train[:, i] = preds_train
        prompt_features_test[:, i] = preds_test
        accs_train[i] = acc_train

    a = np.argsort(accs_train.flatten())[::-1]
    return (
        prompt_features_train[:, a],
        prompt_features_test[:, a],
        np.array(prompts)[a].tolist(),
    )
