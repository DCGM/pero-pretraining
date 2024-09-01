import torch
import pickle
import numpy as np

from pero_pretraining.models.helpers import create_pero_vgg_encoder
from pero_pretraining.autoencoders.model import init_model as init_autoencoder_model


def init_model(model_definition, checkpoint_path, device):
    if model_definition == "pero_vgg":
        model = create_pero_vgg_encoder()

    elif type(model_definition) == str:
        import json
        model_definition = json.loads(model_definition)
        model = init_autoencoder_model(model_definition)

    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.eval()
    model = model.to(device)

    return model


def init_dataset(lmdb_path, lines_path, batch_size):
    dataset = Dataset(lmdb_path=lmdb_path, lines_path=lines_path, augmentations=None, pair_images=False)
    batch_creator = BatchCreator()
    dataloader = create_dataloader(dataset, batch_creator=batch_creator, batch_size=batch_size, shuffle=False)

    return dataloader

def load_pickle(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


def save_pickle(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def save_numpy(data, path):
    with open(path, "wb") as f:
        np.save(f, data)


def save_labels(data, path):
    with open(path, 'w') as f:
        for line_id, line_labels in data.items():
            f.write(f"{line_id} {' '.join([str(label) for label in line_labels])}\n")
