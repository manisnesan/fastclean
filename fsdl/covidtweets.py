from pathlib import Path
from fastai.text.all import *
from fastai.callback.wandb import *
from fastcore.basics import AttrDict

from typing import List

import re, click, json
import pandas as pd
import wandb
import logging

MODELS = "./models"

config = dict(
    SEED=42,
    lm_epoch=2,
    lm_lr=4e-2,
    lm_encoder_path=f"{MODELS}/awd_lstm_fine_tuned_enc",
    bs=64,
)

# Wandb init
wandb.init(project="fsdl-noisylabel-covidtweets", config=config)
cfg = AttrDict(wandb.config)

# Logging init
# Basic Logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def extract_hashtag(text):
    hash_tags = re.findall(r"#(\S+)", text)
    if hash_tags:
        return " ".join(hash_tags)
    else:
        return "NA"


def remove_stop_add_hashtag(df):
    stop_words = ["HTTPURL", "@USER"]

    df["Text"] = df["Text"].apply(
        lambda x: " ".join([each for each in x.split(" ") if each not in stop_words])
    )
    df["HashTag"] = df.Text.apply(extract_hashtag)
    return df


def clean_text(word):

    # Remove urls
    urls = re.findall(r"https?://\S+", word)
    if len(urls) > 0:
        word = ""
        return word

    return word.strip()


def load_data(path):
    covid = path
    # Load all training, dev, test and private test set.
    train_df = pd.read_csv(covid / "train.tsv", delimiter="\t")
    valid_df = pd.read_csv(
        covid / "valid.tsv", names=["Id", "Text", "Label"], delimiter="\t"
    )
    test_df = pd.read_csv(
        covid / "test.tsv", names=["Id", "Text", "Label"], delimiter="\t"
    )
    private_test_df = pd.read_csv(
        covid / "unlabeled_test_with_noise.tsv", names=["Id", "Text"], delimiter="\t"
    )

    return [train_df, valid_df, test_df, private_test_df]


def prepare(all_df):
    stop_words = ["HTTPURL", "@USER"]
    all_df["Text"] = all_df["Text"].apply(
        lambda x: " ".join([each for each in x.split(" ") if each not in stop_words])
    )


@click.command()
@click.argument("data_path", default="../data/covid", type=click.Path())
@click.argument("is_lm", default=True, type=click.BOOL)
def run(data_path, is_lm):

    logging.info(f"Default config : {json.dumps(config)}")  

    # Loading the data
    logging.info('Loading the data...')
    data: List[pd.DataFrame] = load_data(data_path)
    train_df = data[0]
    valid_df = data[1]

    # Process data
    train_df["is_valid"] = False
    valid_df["is_valid"] = True
    train_val = pd.concat([train_df, valid_df])

    all_df = pd.concat(data)
    all_df = remove_stop_add_hashtag(all_df)

    # Language Model
    if is_lm:
        dls_lm = TextDataLoaders.from_df(
            all_df, seed=cfg.SEED, text_col=["HashTag", "Text"], is_lm=True
        )
        lang_learn = language_model_learner(dls_lm, AWD_LSTM, metrics=Perplexity())
        lang_learn.fine_tune(cfg.lm_epoch, 4e-2)
        lang_learn.save_encoder(cfg.lm_encoder_path)
    else:
        # dls_lm = torch.load('SAVED_DATALOADER')
        # load the saved language learner
        pass

    # Classifier
    dls_cls = DataBlock(
        blocks=(
            TextBlock.from_df(
                text_cols=["HashTag", "Text"], is_lm=False, vocab=dls_lm.vocab
            ),
            CategoryBlock,
        ),
        get_x=ColReader("text"),
        get_y=ColReader("Label"),
        splitter=ColSplitter(col="is_valid"),
    ).dataloaders(train_val, bs=cfg.bs)
    learn = text_classifier_learner(dls_cls, AWD_LSTM, metrics=[accuracy, F1Score()], loss_func=CrossEntropyLossFlat())
    learn.load_encoder(cfg.lm_encoder_path)
    learn.fine_tune(1)

    learn.save('base')
