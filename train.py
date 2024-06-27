import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn

from utils import count_model_parameters

PARALLEL = True


class Trainer(object):
    """Training Helper Class"""

    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        if PARALLEL:
            self.device = torch.device("cuda")
        else:
            self.device = device

    def setTeacher(self, model_teacher):
        self.model_teacher = model_teacher

    def printModelInfo(self, path):
        total_params = sum(
            p.numel() for p in self.model_teacher.parameters() if p.requires_grad
        )
        print(f"Total params of teacher model: {total_params}")
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total params of std model: {total_params}")

    def pretrain(
        self,
        func_loss,
        func_forward,
        func_evaluate,
        data_loader_train,
        data_loader_test,
        model_file=None,
    ):
        """Train Loop"""
        self.load(model_file)
        model = self.model.to(self.device)
        if PARALLEL:
            model = nn.DataParallel(model)

        global_step = 0
        best_loss = 1e6
        model_best = model.state_dict()

        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return
            loss_eva = self.run(func_forward, func_evaluate, data_loader_test)
            print(
                "Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f"
                % (
                    e + 1,
                    self.cfg.n_epochs,
                    loss_sum / len(data_loader_train),
                    loss_eva,
                )
            )
            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)

        model.load_state_dict(model_best)
        print("The Total Epoch have been reached.")

    def distill(
        self,
        func_loss,
        func_forward,
        func_evaluate,
        data_loader_train,
        data_loader_test,
        teacher_model_file=None,
        config=None,
    ):
        self.load_teacher_file(teacher_model_file)
        self.load(self.save_path)
        total_params = sum(
            p.numel() for p in self.model_teacher.parameters() if p.requires_grad
        )
        print(f"Total teacher Params: {total_params}")
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total std Params: {total_params}")

        model_std = self.model.to(self.device)
        model_teacher = self.model_teacher.to(self.device)

        global_step = 0
        best_loss = 1e6
        model_best = model_std.state_dict()

        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0
            time_sum = 0.0
            self.model_teacher.eval()
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss, loss_list = func_loss(model_std, model_teacher, batch, config)
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                global_step += 1
                loss_temp = loss.item()
                loss_sum += loss_temp

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return
            print(
                f"Epoch {e + 1}/{self.cfg.n_epochs} :best_loss {best_loss}. Average Loss {loss_sum / len(data_loader_train):5.4f}.  Loss {loss_temp:5.4f} Loss Detail: {' '.join([f'{x.item():.4f}' for x in loss_list])}"
            )

            if loss_temp < best_loss:
                best_loss = loss_temp
                model_best = copy.deepcopy(model_std.state_dict())
                print(
                    f"[Distilling] Reach a best result, save the model to ->${self.save_path}"
                )
                self.save(0)

        model_std.load_state_dict(model_best)
        print("[Distilling] The Total Epoch have been reached.")

    def run(
        self,
        func_forward,
        func_evaluate,
        data_loader,
        model_file=None,
        load_self=False,
    ):
        """Evaluation Loop"""
        self.model.eval()
        self.load(model_file, load_self=load_self)
        model = self.model.to(self.device)
        if PARALLEL:
            model = nn.DataParallel(model)

        results = []
        labels = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                start_time = time.time()
                result, label = func_forward(model, batch)
                time_sum += time.time() - start_time
                results.append(result)
                labels.append(label)
        if func_evaluate:
            return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
        else:
            return torch.cat(results, 0).cpu().numpy()

    def train(
        self,
        func_loss,
        func_forward,
        func_evaluate,
        data_loader_train,
        data_loader_test,
        data_loader_vali,
        model_file=None,
        load_self=False,
    ):
        """Train Loop"""
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if PARALLEL:
            model = nn.DataParallel(model)

        global_step = 0
        vali_acc_best = 0.0
        best_stat = None
        model_best = model.state_dict()
        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print("The Total Steps have been reached.")
                    return
            train_acc, train_f1 = self.run(
                func_forward, func_evaluate, data_loader_train
            )
            test_acc, test_f1 = self.run(func_forward, func_evaluate, data_loader_test)
            vali_acc, vali_f1 = self.run(func_forward, func_evaluate, data_loader_vali)
            print(
                "Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f"
                % (
                    e + 1,
                    self.cfg.n_epochs,
                    loss_sum / len(data_loader_train),
                    train_acc,
                    vali_acc,
                    test_acc,
                    train_f1,
                    vali_f1,
                    test_f1,
                )
            )
            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
        print("Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f" % best_stat)
        print("The Total Epoch have been reached.")

    def load(self, model_file, load_self=False):
        """load saved model or pretrained transformer (a part of model)"""
        if model_file:
            print("Loading the model from", model_file)
            if load_self:
                self.model.load_self(model_file + ".pt", map_location=self.device)
            else:
                self.model.load_state_dict(
                    torch.load(model_file + ".pt", map_location=self.device)
                )

    def load_student_file(self, model_file, load_self=False):
        """load saved model or pretrained transformer (a part of model)"""
        if model_file:
            print("Loading the std_model from", model_file)
            if load_self:
                self.model.load_self(model_file + ".pt", map_location=self.device)
            else:
                self.model.load_state_dict(
                    torch.load(model_file + ".pt", map_location=self.device)
                )

    def load_teacher_file(self, model_file, load_self=False):
        """load saved model or pretrained transformer (a part of model)"""
        if model_file:
            print("Loading the teacher_model from", model_file)
            if load_self:
                self.model_teacher.load_self(
                    model_file + ".pt", map_location=self.device
                )
            else:
                self.model_teacher.load_state_dict(
                    torch.load(model_file + ".pt", map_location=self.device)
                )

    def save(self, i=0):
        """save current model"""
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + ".pt")
        else:
            torch.save(self.model.state_dict(), self.save_path + ".pt")
