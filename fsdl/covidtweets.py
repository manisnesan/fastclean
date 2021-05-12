from pathlib import Path
from fastai.text.all import *
from fastai.callback.wandb import *
from fastcore.basics import AttrDict
from cleanlab.pruning import get_noise_indices

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
# wandb.init(project="fsdl-noisylabel-covidtweets", config=config)
# cfg = AttrDict(wandb.config)
cfg = AttrDict(config)

# Logging init
# Basic Logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def extract_hashtag(text):
    hash_tags = re.findall(r"#(\S+)", text)
    if hash_tags:
        return " ".join(hash_tags)
    else:
        return "NA"

def preprocess(text):
    stop_words = ["HTTPURL", "@USER"]
    clean_text = " ".join([each for each in text.split(" ") if each not in stop_words])

    hash_tags = re.findall(r"#(\S+)", clean_text)
    f'{" ".join(hash_tags)} {clean_text}' 

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
@click.option("--mode", default='train', type=click.STRING)
def run(data_path, mode, is_lm):

    if mode == 'predict':
        return predict(text)

    logging.info(f"Default config : {json.dumps(config)}")  

    # Loading the data    
    logging.info('Loading the data...')
    data: List[pd.DataFrame] = load_data(Path(data_path))

    # Evaluation
    if mode == 'eval':
        evaluate(data[2], data_path)
        return

    # Training 
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


def predict(text):

    # load the saved model
    learn = load_learner(f'{MODELS}/{cfg.model_name}')

    clas, clas_idx, probs = learn.predict(tokenize1(text, tok=WordTokenizer()))
    return clas

def evaluate(df, data_path):

    # preprocess test data
    test_df = remove_stop_add_hashtag(df)

    # tokenize 
    tokenized_df = tokenize_df(test_df, text_cols=["HashTag", "Text"], mark_fields=True, tok_text_col='text') #returns a tuple

    # load the saved model
    learn = load_learner(f'{MODELS}/{cfg.model_name}')

    # test dataloader
    test_dl = learn.dls.test_dl(tokenized_df[0])

    # predictions
    result = learn.get_preds(dl=test_dl)

    confidence = torch.max(result[0], axis=1).values
    _, y = learn.dls.valid.vocab
    y_predicted = np.array(y[result[0].argmax(axis=1)])

    test_df['predicted'] = y_predicted
    test_df['confidence'] = confidence

    # metrics
    # _, metric_value = learn.validate(dl=test_dl) #loss, metrics used
    # metrics = {each.name: metric_value[idx] for idx, each in enumerate(learn.metrics)}
    # print(f"Metrics on the test dataset : {metrics}")

    # Noisy Labels on Test Dataframe
    test_ordered_label_errors = get_noise_indices(s=Numericalize(vocab=['INFORMATIVE', 'UNINFORMATIVE'])(test_df.Label).numpy(), 
                                         psx=result[0].numpy(),
                                         prune_method="both", # 'prune_by_noise_rate': works by removing examples with *high probability* of being mislabeled for every non-diagonal in the prune_counts_matrix (see pruning.py).
                                                              #'prune_by_class': works by removing the examples with *smallest probability* of belonging to their given class label for every class.
                                         sorted_index_method='normalized_margin')

    print(test_ordered_label_errors)
    test_df.iloc[test_ordered_label_errors].to_csv(f'{data_path}/noisy_text.csv')


if __name__ == '__main__':
    run()