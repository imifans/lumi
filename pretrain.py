import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import train
from models import BertModel4Pretrain
from utils import (
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


def main(args):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg, training_rate = (
        load_pretrain_data_config(args)
    )
    pipeline = [norm_processor(model_cfg.feature_num), mask_processor(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(
        data, labels, training_rate, seed=train_cfg.seed
    )
    data_set_train = buildDataSet4PreTrain(data_train, pipeline)
    data_set_test = buildDataSet4PreTrain(data_test, pipeline)
    data_loader_train = DataLoader(
        data_set_train, shuffle=True, batch_size=train_cfg.batch_size
    )
    data_loader_test = DataLoader(
        data_set_test, shuffle=False, batch_size=train_cfg.batch_size
    )
    model = BertModel4Pretrain(model_cfg)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_reconstructure = model(mask_seqs, masked_pos)
        loss_lm = criterion(seq_reconstructure, seqs)
        return loss_lm

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, "pretrain_model"):
        trainer.pretrain(
            func_loss,
            func_forward,
            func_evaluate,
            data_loader_train,
            data_loader_test,
            model_file=args.pretrain_model,
        )
    else:
        trainer.pretrain(
            func_loss,
            func_forward,
            func_evaluate,
            data_loader_train,
            data_loader_test,
            model_file=None,
        )


if __name__ == "__main__":
    mode = "base"
    train_config_name = "pretrain.json"
    args = handle_argv("pretrain_" + mode, train_config_name, mode)
    main(args)
