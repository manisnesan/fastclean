from fsdl.train import load_data, get_dls
import pytest, torch
import fastai, fastcore, nbdev


def test_load_data():
    source, df = load_data()
    assert source
    assert len(df)
    expected = ['path', 'noisy_labels_0', 'noisy_labels_1', 'noisy_labels_5', 'noisy_labels_25', 'noisy_labels_50', 'is_valid']
    assert expected == list(df.columns)

def test_get_dls():
    source, df = load_data()
    dls = get_dls(df, pref=source, size=224)
    b = dls.one_batch()
    assert 2 == len(b)
    assert torch.Size([64, 3, 224, 224]) == b[0].shape
    assert torch.Size([64]) == b[1].shape

    expected = ['English springer', 'French horn', 'cassette player', 'chain saw', 'church', 'garbage truck', 'gas pump', 'golf ball', 'parachute', 'tench']
    assert expected == dls.valid.vocab
