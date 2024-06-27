import os

import numpy as np
from torch import nn
from torch.utils.data import DataLoader


import train
from config import load_dataset_label_names
from models import BertModel4Pretrain, BertStudentModel
from plot import plot_reconstruct_sensor, plot_embedding
from utils import (
    buildDataSet4PreTrain,
    load_pretrain_data_config,
    get_device,
    handle_argv,
    norm_processor,
    IMUDataset,
)


def fetch_distill_setup(args, output_embed):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg, training_rate = (
        load_pretrain_data_config(args)
    )
    pipeline = [norm_processor(model_cfg.feature_num)]
    data_set = IMUDataset(data, labels, pipeline=pipeline)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
    model = BertStudentModel(
        model_cfg,
        output_embed=output_embed,
        reduction_factor=train_cfg.reduction_factor,
    )
    criterion = nn.MSELoss(reduction="none")
    return data, labels, data_loader, model, criterion, train_cfg


def fetch_setup(args, output_embed):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg, training_rate = (
        load_pretrain_data_config(args)
    )
    pipeline = [norm_processor(model_cfg.feature_num)]
    data_set = IMUDataset(data, labels, pipeline=pipeline)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
    model = BertModel4Pretrain(model_cfg, output_embed=output_embed)
    criterion = nn.MSELoss(reduction="none")
    return data, labels, data_loader, model, criterion, train_cfg


def generate_embedding_or_output(args, save=False, output_embed=True, distill=False):
    if distill:
        data, labels, data_loader, model, criterion, train_cfg = fetch_distill_setup(
            args, output_embed
        )
    else:
        data, labels, data_loader, model, criterion, train_cfg = fetch_setup(
            args, output_embed
        )

    optimizer = None
    trainer = train.Trainer(
        train_cfg, model, optimizer, args.save_path, get_device(args.gpu)
    )

    def func_forward(model, batch):
        seqs, label = batch
        embed = model(seqs)
        return embed, label

    path = args.save_path if distill else args.pretrain_model
    output = trainer.run(func_forward, None, data_loader, path)
    prefix = "embed_distilled_" if distill else "embed_"
    if save:
        save_name = (
            prefix
            + args.model_file.split(".")[0]
            + "_"
            + args.dataset
            + "_"
            + args.dataset_version
        )
        path = os.path.join("embed", save_name + ".npy")
        np.save(path, output)
        print(f"[Profile] saved in {path}... ")
    return data, output, labels


def load_embedding_label(model_file, dataset, dataset_version):
    embed_name = "embed_" + model_file + "_" + dataset + "_" + dataset_version
    label_name = "label_" + dataset_version
    embed = np.load(os.path.join("embed", embed_name + ".npy")).astype(np.float32)
    labels = np.load(os.path.join("dataset", dataset, label_name + ".npy")).astype(
        np.float32
    )
    return embed, labels


def load_embedding_distilled_label(model_file, dataset, dataset_version):
    embed_name = "embed_distilled_" + model_file + "_" + dataset + "_" + dataset_version
    label_name = "label_" + dataset_version
    embed = np.load(os.path.join("embed", embed_name + ".npy")).astype(np.float32)
    labels = np.load(os.path.join("dataset", dataset, label_name + ".npy")).astype(
        np.float32
    )
    print(f"[Loading]  Loading Embeddings Distilled from {embed_name}")
    return embed, labels


if __name__ == "__main__":
    save = True
    mode = "base"
    args = handle_argv("pretrain_" + mode, "pretrain.json", mode)
    data, output, labels = generate_embedding_or_output(
        args=args, output_embed=True, save=save
    )
