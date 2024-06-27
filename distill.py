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


def distill(args):

    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg, _ = (
        load_pretrain_data_config(args)
    )

    data_loader_train, data_loader_test = get_data_loader(args)

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

    def func_loss(model_s, model_t, batch, config):
        hidden_distill_factor = config["hidden_distill_factor"]
        attention_distill_factor = config["attention_distill_factor"]
        beta_distill_factor = config["beta_distill_factor"]
        gamma_distill_factor = config["gamma_distill_factor"]
        mlm_distill_factor = config["mlm_distill_factor"]

        mask_seqs, masked_pos, seqs = batch
        (seq_t, featureMap_t, attMap_t) = model_t(mask_seqs, masked_pos, distill=True)
        (seq_s, featureMap_s, attMap_s) = model_s(mask_seqs, masked_pos, distill=True)

        extra_loss1 = torch.zeros(1).to(device)
        extra_loss2 = torch.zeros(1).to(device)
        extra_loss3 = torch.zeros(1).to(device)
        extra_loss4 = torch.zeros(1).to(device)
        MLM_loss = torch.zeros(1).to(device)

        # 1. feature_map_transfer_loss
        if hidden_distill_factor != 0:
            total = torch.zeros(1, device=device)
            for tensor_s, tensor_t in zip(featureMap_s, featureMap_t):
                mse_loss = torch.mean((tensor_s - tensor_t) ** 2)
                total += mse_loss
            extra_loss1 = total * hidden_distill_factor / layers_num
            extra_loss1 = extra_loss1.to(device)

        # 2. attention_transfer_loss
        if attention_distill_factor != 0:
            total = torch.zeros(1, device=device)
            for tensor_s, tensor_t in zip(attMap_s, attMap_t):
                with torch.no_grad():
                    teacher_attention_prob = F.softmax(tensor_t, dim=-1)
                student_attention_log_prob = F.log_softmax(tensor_s, dim=-1)
                #  KL
                kl_temp = -torch.sum(
                    teacher_attention_prob.detach() * student_attention_log_prob, dim=-1
                )
                kl_temp_mean = torch.mean(kl_temp)
                total += kl_temp_mean
            extra_loss2 = total * attention_distill_factor / layers_num
            extra_loss2 = extra_loss2.to(device)

        # 3. mean_transfer_loss
        if beta_distill_factor != 0:
            total = torch.zeros(1, device=device)
            for t, s in zip(featureMap_t, featureMap_s):
                t_mean = torch.mean(t, dim=-1, keepdim=True)
                s_mean = torch.mean(s, dim=-1, keepdim=True)
                total += torch.mean((t_mean.detach() - s_mean) ** 2)
            extra_loss3 = total * beta_distill_factor / layers_num
            extra_loss3 = extra_loss3.to(device)

        # 4. variance_transfer_loss
        if gamma_distill_factor != 0:
            total = torch.zeros(1, device=device)
            for t, s in zip(featureMap_t, featureMap_s):
                t_mean = torch.mean(t, dim=-1, keepdim=True)
                s_mean = torch.mean(s, dim=-1, keepdim=True)
                t_variance = torch.mean((t - t_mean) ** 2, dim=-1, keepdim=True)
                s_variance = torch.mean((s - s_mean) ** 2, dim=-1, keepdim=True)
                total += torch.mean(torch.abs(t_variance.detach() - s_variance))
            extra_loss4 = total * gamma_distill_factor / layers_num
            extra_loss4 = extra_loss4.to(device)

        # 5. MLM loss
        if mlm_distill_factor != 0:
            MLM_loss = criterion(seq_s, seqs)
            MLM_loss = MLM_loss.to(device)

        # total
        loss = extra_loss1 + extra_loss2 + extra_loss3 + extra_loss4 + MLM_loss
        loss_list = [extra_loss1, extra_loss2, extra_loss3, extra_loss4, MLM_loss]
        loss = loss.to(device)

        return loss, loss_list

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    print(f"[Distilling] save a empty std -> {args.save_path}")
    torch.save(model_s.state_dict(), args.save_path + ".pt")
    trainer.printModelInfo(args.pretrain_model)
    # extra loss
    print("[Distilling]: start to attention distilling")

    config = {
        "hidden_distill_factor": 100,
        "attention_distill_factor": 1,
        "beta_distill_factor": 5000,
        "gamma_distill_factor": 5,
        "mlm_distill_factor": 0,
    }

    trainer.distill(
        func_loss,
        func_forward,
        func_evaluate,
        data_loader_train,
        data_loader_test,
        teacher_model_file=args.pretrain_model,
        config=config,
    )

    config = {
        "hidden_distill_factor": 0,
        "attention_distill_factor": 0,
        "beta_distill_factor": 0,
        "gamma_distill_factor": 0,
        "mlm_distill_factor": 1,
    }
    trainer.distill(
        func_loss,
        func_forward,
        func_evaluate,
        data_loader_train,
        data_loader_test,
        teacher_model_file=args.pretrain_model,
        config=config,
    )


# python distill.py v1 uci 20_120 -s limu_v1
if __name__ == "__main__":
    mode = "base"
    args = handle_argv("distill_" + mode, "distill.json", mode)
    distill(args)
