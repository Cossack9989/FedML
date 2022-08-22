import torch
from torch import nn

from ....core.alg_frame.client_trainer import ClientTrainer
import copy
import joblib
import logging
import numpy as np


class MyModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, class_weight, device, args):
        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        # epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                
                pred_res = model(x)
                # if isinstance(pred_res, tuple) and len(pred_res) == 2:
                #     middle_preds, log_probs = pred_res[0], pred_res[1]
                # else:
                log_probs = pred_res
                loss = criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
            # epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info(
            #     "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
            #         self.id, epoch, sum(epoch_loss) / len(epoch_loss)
            #     )
            # )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
                "test_correct": 0, 
                "test_loss": 0, 
                "test_total": 0,
                "test_recall": {},
                "test_precision": {},
                "_test_recall": {},
                "_test_precision": {}
            }
        typ_list = []
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred_res = model(x)
                # if isinstance(pred_res, tuple) and len(pred_res) == 2:
                #     pred = pred_res[1]
                # else:
                pred = pred_res
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                typ_list += target.tolist()
                for typ in list(set(target.tolist())):
                    if typ not in metrics["_test_recall"].keys():
                        metrics["_test_recall"][typ] = {
                            "numerator": 0,
                            "denominator": 0,
                        }
                    if typ not in metrics["_test_precision"].keys():
                        metrics["_test_precision"][typ] = {
                            "numerator": 0,
                            "denominator": 0,
                        }
                    positive_idxs = np.where(target.cpu().numpy() == typ)[0]
                    true_positive = np.sum(target.cpu().numpy()[positive_idxs] == predicted.cpu().numpy()[positive_idxs])
                    all_actually_positive = len(positive_idxs)
                    all_predicted_positive = len(np.where(predicted.cpu().numpy() == typ)[0])
                    metrics["_test_recall"][typ]["numerator"] += true_positive
                    metrics["_test_precision"][typ]["numerator"] += true_positive
                    metrics["_test_recall"][typ]["denominator"] += all_actually_positive
                    metrics["_test_precision"][typ]["denominator"] += all_predicted_positive
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        for typ in list(set(typ_list)):
            metrics["test_recall"][typ] = (metrics["_test_recall"][typ]["numerator"] + 1e-13) / (metrics["_test_recall"][typ]["denominator"] + 1e-13)
            metrics["test_recall"][typ] = 0 if metrics["test_recall"][typ] < 1e-13 else metrics["test_recall"][typ]
            metrics["test_precision"][typ] = (metrics["_test_precision"][typ]["numerator"] + 1e-13) / (metrics["_test_precision"][typ]["denominator"] + 1e-13)
            metrics["test_precision"][typ] = 0 if metrics["test_precision"][typ] < 1e-13 else metrics["test_precision"][typ]
        return metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return False
