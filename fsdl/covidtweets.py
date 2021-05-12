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
    bs=64,
    clf_epoch=10,
    clf_lr=8e-4,
    lm_epoch=3,
    lm_lr=4e-2,
    lm_encoder_name=f"awd_lstm_fine_tuned_enc",
    model_name='covid_base.pkl',
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


def load_data(path: Path):
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
@click.argument("data_path", default="./data/covid", type=click.Path())
@click.argument("is_lm", default=True, type=click.BOOL)
def run(data_path, is_lm):

    logging.info(f"Default config : {json.dumps(config)}")  

    # Loading the data
    logging.info('Loading the data...')
    data: List[pd.DataFrame] = load_data(Path(data_path))
    train_df = data[0]
    valid_df = data[1]

    # Process data
    train_df["is_valid"] = False
    valid_df["is_valid"] = True
    train_val = pd.concat([train_df, valid_df])
    train_val = remove_stop_add_hashtag(train_val)

    all_df = pd.concat(data)
    all_df = remove_stop_add_hashtag(all_df)

    dls_lm = TextDataLoaders.from_df(
            all_df, seed=cfg.SEED, text_col=["HashTag", "Text"], is_lm=True
        )
    # Language Model
    if is_lm:
        logging.info('Training the language model ...')
        lang_learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[Perplexity(), accuracy], model_dir=MODELS,
                                            cbs=[WandbCallback(log_preds=False, log_model=False), 
                                                SaveModelCallback(fname=f'language_model')])
        lang_learn.fine_tune(cfg.lm_epoch, cfg.lm_lr)
        logging.info(f'Saving the encoder from language model with the name {cfg.lm_encoder_name}')
        lang_learn.save_encoder(cfg.lm_encoder_name)
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
    
    logging.info('Training the classifier ...')
    learn = text_classifier_learner(dls_cls, AWD_LSTM, metrics=[error_rate, F1Score()], loss_func=CrossEntropyLossFlat(), model_dir=MODELS,
                                            cbs=[WandbCallback(log_preds=False, log_model=False), 
                                                 SaveModelCallback(fname='classifier')])
    learn.load_encoder(cfg.lm_encoder_name)
    learn.fine_tune(cfg.clf_epoch, cfg.clf_lr)

    logging.info(f'Saving the classifier model {cfg.model_name}')
    learn.export(f'{MODELS}/{cfg.model_name}')

if __name__ == '__main__':
    run()