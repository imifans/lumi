import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score

from plot import plot_matrix
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import BertModel4Pretrain, BertStudentModel, TransformerWithBottleneck
import torch.nn.functional as F

from utils import (
    get_data_loader,
    set_seeds,
    get_device,
    buildDataSet4PreTrain,
    handle_argv,
    load_pretrain_data_config,
    prepare_classifier_dataset,
    prepare_pretrain_dataset,
    norm_processor,
    mask_processor,
)


def stat_acc_f1(label, results_estimated):
    # label = np.concatenate(label, 0)
    # results_estimated = np.concatenate(results_estimated, 0)
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average="macro")
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1


def stat_acc_f1_dual(label, results_estimated):
    label = np.concatenate(label, 0)
    results_estimated = np.concatenate([t[1] for t in results_estimated], 0)
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average="macro")
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1


def stat_results(label, results_estimated):
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average="macro")
    acc = np.sum(label == label_estimated) / label.size
    matrix = metrics.confusion_matrix(label, label_estimated)  # , normalize='true'
    return acc, matrix, f1


def stat_acc_f1_tpn(label, label_estimated, task_num=5, threshold=0.5):
    label_new = []
    label_estimated_new = []
    for i in range(label.size):
        if label[i] == 0:
            label_new.append(np.zeros((task_num, 1)))
            label_estimated_new_temp = np.zeros((task_num, 1))
            label_estimated_new_temp[label_estimated[i, :] > threshold] = 1
            label_estimated_new.append(label_estimated_new_temp)
        else:
            label_new.append(np.ones((1, 1)))
            label_estimated_new_temp = np.zeros((1, 1))
            label_estimated_new_temp[label_estimated[i, label[i] - 1] > threshold] = 1
            label_estimated_new.append(label_estimated_new_temp)
    label_new = np.concatenate(label_new, 0)[:, 0]
    label_estimated_new = np.concatenate(label_estimated_new, 0)[:, 0]
    f1 = f1_score(label_new, label_estimated_new, average="macro")
    acc = np.sum(label_new == label_estimated_new) / label_new.size
    return acc, f1


def pring_models(arg):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg, _ = (
        load_pretrain_data_config(args)
    )

    model_t = BertModel4Pretrain(model_cfg)
    model_s = BertStudentModel(model_cfg, reduction_factor=train_cfg.reduction_factor)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model_s.parameters(), lr=train_cfg.lr)

    layers_num = model_cfg.n_layers
    device = get_device(args.gpu)
    trainer = train.Trainer(
        train_cfg,
        model_s,
        optimizer,
        args.save_path,
        device,
    )
    trainer.setTeacher(model_t)
    trainer.printModelInfo(args.pretrain_model)


if __name__ == "__main__":
    # eg: python statistic.py v1 uci 20_120 -s limu_v1
    mode = "base"
    train_config_name = "distill.json"
    args = handle_argv("distill_" + mode, train_config_name, mode)
    pring_models(args)
