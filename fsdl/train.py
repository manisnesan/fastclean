# Script that performs training of noisy imagenette dataset 
# and returns the predictions and confidence scores of the 
# validation dataset.

from fastai.vision.all import *
from fastai.callback.wandb import *
from cleanlab.pruning import get_noise_indices
from pathlib import Path

import click
import logging
import json
import wandb
# TODO
# [] Refactor the FILENAME used for loading the data

MODELS = './models'
# Wandb init
wandb.init(project="fsdl-noisylabel")

# Default Config
config = dict(
    base_lr=1e-3,
    epochs=5,
    freeze_epochs=3,
    seed=42,
)
wandb.config.update(config)

# Ensuring the script is reproducible
set_seed(42, reproducible=True)

# Basic Logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

lbl_dict = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute'
)
lbl_dict_inv = {v: k for k, v in lbl_dict.items()}
def get_inverse_transform(vocab):
  return L(vocab).map(lbl_dict_inv)

def get_dls(df, pref, noice_pct=5, size=128, soft_targets=False):
    # Return dataloaders from path provided using DataBlock API specification
    if soft_targets:
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                            get_x=ColReader('path', pref=pref),
                            get_y=Pipeline([ColReader(f'soft_targets'), lbl_dict.__getitem__]),
                            splitter=ColSplitter(), #uses the bool value in is_valid column on the dataframe to identify the validation set (without any noise).
                            item_tfms=[RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)],
                            batch_tfms=Normalize.from_stats(*imagenet_stats))

    else:
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                        get_x=ColReader('path', pref=pref),
                        get_y=Pipeline([ColReader(f'noisy_labels_{noice_pct}'), lbl_dict.__getitem__]),
                        splitter=ColSplitter(), #uses the bool value in is_valid column on the dataframe to identify the validation set (without any noise).
                        item_tfms=[RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)],
                        batch_tfms=Normalize.from_stats(*imagenet_stats))
    dls = dblock.dataloaders(df)
    return dls


def train(dls: DataLoaders, filename: str='export.pkl') -> Learner:
    '''
    Train the resnet18 model using the provided dataloaders and save the model using 'filename'
    ''' 
    logging.info('Training the resnet model...')
    
    learn = cnn_learner(dls, resnet34, metrics=[accuracy, RocAuc()],
                        loss_func=LabelSmoothingCrossEntropyFlat(),
                        cbs=[WandbCallback(log_preds=False, log_model=False), SaveModelCallback(fname=filename)])

    #learn.fit_one_cycle(1, 1e-3)
    learn.fine_tune(epochs=5, base_lr=1e-3, freeze_epochs=3)

    logging.info(f'Saving the model with name {filename}')
    learn.export(f'{MODELS}/{filename}')

    # Marking the run as completed.
    wandb.finish()
    return learn


def load_data(url=URLs.IMAGENETTE):
    "Load the data from the desired url and return path downloaded along with the dataframe"
    source = untar_data(url)
    df: pd.DataFrame = pd.read_csv(source/'noisy_imagenette.csv')
    return source, df


@click.command()
@click.argument("output_filepath", type=click.Path())
@click.option("--noice_pct", default=5, help="Noice percent level")
def main(output_filepath, noice_pct):
    '''
    Evaluation script and write the noisy indices to a separate csv file
    '''

    logging.info(f"Default config : {json.dumps(config)}")

    # Loading the data
    logging.info('Loading the data...')
    source, df = load_data(URLs.IMAGENETTE)

    # Default DataLoaders using 5 as noise_percent
    dls: DataLoaders = get_dls(df, noice_pct=noice_pct, pref=source, size=224)

    learn: Learner = train(dls, f'resnet34_5ep_3frzep_1e_3_{noice_pct}np.pkl')


    # Predict using a single image file
    # learn = load_learner('export.pkl')
    # learn.predict('TEST_IMAGE_FILE')

    # Determine the noisy indices on training/validation or an unknown test dataset
    # Training 
    logging.info('Getting the predictions for training data...')
    train_preds = learn.get_preds(ds_idx=0, with_decoded=True)

    # Add Predictions & Confidence Score
    decoded_train_preds = get_inverse_transform(L(list(train_preds[2])).map(dls.vocab))
    
    # Add confidence & confidenceRange
    confidence = torch.max(train_preds[0], axis=-1).values

    # Noisy indices from training dataset.
    train_ordered_label_errors = get_noise_indices(s=train_preds[1].numpy(), #targets
                             psx=train_preds[0].numpy(),#predictions_prob
                             sorted_index_method='normalized_margin')

    # Actual Noise in the training dataset
    train_df = df[df.is_valid == False].copy()
    print("We found {} label errors in the training dataset of size {}.".format(len(train_ordered_label_errors), len(train_df)))

    train_df['predictions'] = np.array(decoded_train_preds)
    train_df['confidence'] = confidence
    noisy_train = train_df.iloc[train_ordered_label_errors]

    PREDICTIONS_NAME = f'noisy{noice_pct}_train_predictions.csv'
    logging.info(f'Saving the noisy training data with predictions and confidence at {output_filepath}/{PREDICTIONS_NAME}')
    noisy_train.to_csv(f'{output_filepath}/{PREDICTIONS_NAME}', index=False)
    
if __name__ == '__main__':
    main()
