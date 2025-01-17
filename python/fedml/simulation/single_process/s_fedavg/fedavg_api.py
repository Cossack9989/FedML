import copy
import os
import time
import random
import joblib
import secrets
import getpass
import numpy as np
import torch
import wandb
import logging

from hashlib import sha1
from itertools import combinations
from scipy.spatial import distance
from collections import Counter
from sklearn.metrics import f1_score, recall_score, precision_score
from .client import Client
from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


class S_FedAvgAPI(object):

    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
            valid_data_in_aggregator,
            alpha,
            beta,
            sampling_filter,
            sv_approaching,
            score_method,
            target_label
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.global_valid_data = valid_data_in_aggregator
        self.arguments = {
            "alpha": alpha,
            "beta": beta
        }
        self.sv_approaching = sv_approaching
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.sampling_filter = sampling_filter
        self.score = score_method
        self.target_label = target_label

        logging.info("model = {}".format(model))
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else:
            # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
        self.model_trainer = model_trainer
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.model_trainer,
        )
        self.seed = secrets.SystemRandom().randint(0x0, 0xffffffff)

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        # if getpass.getuser() != 'c0ss4ck':
        #     exit(0)
        # if sha1(os.environ["FEDML_KEY"].encode('latin-1')).hexdigest() != "f2141ae4f4176b51e3679760e6233b870143c5f2":
        #     exit(0)
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
                self.global_valid_data,
                self.calc_class_weight(self.train_data_local_dict[client_idx])
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    # 每个节点每个class的权重为该节点数据总量 /（该节点内有几种class * 该节点内该class样本数量）
    def calc_class_weight(self, train_data):
        y = []
        for batch in train_data:
            y += batch[1].numpy().tolist()
        class_weight = []
        if self.args.dataset in ["mit-bih"]:
            class_num = 5
        elif self.args.dataset in ["mnist", "cifar10"]:
            class_num = 10
        elif self.args.dataset in ["kag-nih"]:
            class_num = 15
        elif self.args.dataset in ["bimcv-covid"]:
            class_num = 2
        else:
            class_num = 0
            logging.info("not support")
            exit(0)
        for _class in range(class_num):
            if _class not in y:
                class_weight.append(0)
            else:
                weight = len(y) / (len(np.unique(y)) * Counter(y)[_class])
                class_weight.append(weight)
        # class_weight = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        return torch.tensor(np.array(class_weight), dtype=torch.float)

    def isApproached(self, d_list, approaching_limit=0.005):
        if len(d_list) >= self.args.client_num_per_round ** 2:
            return False
        if len(d_list) <= self.args.client_num_per_round:
            return True
        for d in d_list[-3:]:
            if d >= approaching_limit:
                return True
        return False

    def train(self):

        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()

        K = self.args.client_num_in_total
        alpha = self.arguments["alpha"]
        beta = self.arguments["beta"]
        phi_dict = {}
        res_dict = {}
        sv_dict = {}
        client_dict = {}
        time_dict = {}
        phi = [1 / K] * K
        sv = [(1 - alpha) / (K * beta)] * K

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            client_dict[round_idx] = {}

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round, phi, self.sampling_filter
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                    self.calc_class_weight(self.train_data_local_dict[client_idx])
                )

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                # self.logging.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            if self.sv_approaching:
                t_head = time.time()
                sv_current = [0.0] * self.args.client_num_per_round
                sv_approached = [0.0] * self.args.client_num_per_round
                sv_last_round = [0.0] * self.args.client_num_per_round
                d = []
                approaching_cnt = 0
                client_indexs = list(range(self.args.client_num_per_round))
                np.random.seed((self.seed * int(time.time())) & 0xffffffff)
                while self.isApproached(d_list=d):
                    np.random.shuffle(client_indexs)
                    used_value = 0
                    if approaching_cnt != 0:
                        sv_last_round = copy.deepcopy(sv_current)
                    for tmp_permutation_idx in range(1, self.args.client_num_per_round):
                        tmp_w_locals = w_locals[:tmp_permutation_idx]
                        tmp_model_trainer = copy.deepcopy(self.model_trainer)
                        tmp_w_global = self._aggregate(tmp_w_locals)
                        tmp_model_trainer.set_model_params(tmp_w_global)
                        tmp_m = self._valid_test_on_aggregator(
                            tmp_model_trainer.model, self.global_valid_data, self.device)
                        if isinstance(self.target_label, int):
                            if self.score == "F1":
                                ap = tmp_m["F1"] - used_value
                                used_value = tmp_m["F1"]
                            elif self.score in ["Sensitivity", "Recall", "TPR", "tpr"]:
                                ap = tmp_m["Recall"] - used_value
                                used_value = tmp_m["Recall"]
                            elif self.score in ["Precision", "PPV", "ppv"]:
                                ap = tmp_m["Precision"] - used_value
                                used_value = tmp_m["Precision"]
                            else:
                                ap = (tmp_m["test_correct"] / tmp_m["test_total"]) - used_value
                                used_value = (tmp_m["test_correct"] / tmp_m["test_total"])
                        else:
                            ap = (tmp_m["test_correct"] / tmp_m["test_total"]) - used_value
                            used_value = (tmp_m["test_correct"] / tmp_m["test_total"])
                        sv_current[tmp_permutation_idx] = \
                            (approaching_cnt * sv_current[tmp_permutation_idx] + ap) / (approaching_cnt + 1)

                    if approaching_cnt != 0:
                        d.append(distance.euclidean(sv_last_round, sv_current))
                    approaching_cnt += 1

                sv_approached = copy.deepcopy(sv_current)
                logging.info(f"Approaching: {sv_approached}")
                t_tail = time.time()
                t_cost = t_tail - t_head
                logging.info(f"[cost {t_cost}]")
                for idx in range(self.args.client_num_per_round):
                    client_idx = client_indexes[idx]
                    tmp_w_single = self._aggregate(copy.deepcopy([w_locals[idx]]))
                    tmp_model_trainer = copy.deepcopy(self.model_trainer)
                    tmp_model_trainer.set_model_params(tmp_w_single)
                    tmp_m_single = self._valid_test_on_aggregator(
                        tmp_model_trainer.model, self.global_valid_data, self.device
                    )
                    client_dict[round_idx][client_idx] = tmp_m_single["test_correct"] / tmp_m_single["test_total"]
                    sv[client_idx] = sv_approached[idx]
                    phi[client_idx] = alpha * phi[client_idx] + beta * sv[client_idx]
                    logging.info(f"Client {client_idx} sv={sv[client_idx]} phi={phi[client_idx]}")

            else:
                t_head = time.time()
                for idx, client in enumerate(self.client_list):
                    # calculate shapley
                    # logging.info(f"calculating shapley of client {idx}")
                    tmp_w_locals = copy.deepcopy(w_locals)
                    tmp_client_idx_list = list(range(len(self.client_list)))
                    del tmp_client_idx_list[idx]

                    ap = 0
                    cnt = 0
                    total_combs = []
                    for left_client_num in range(1, len(self.client_list)):
                        combs = list(combinations(tmp_client_idx_list, left_client_num))
                        cnt += len(combs)
                        total_combs += list(combinations(tmp_client_idx_list, left_client_num))

                        for client_set in combs:
                            client_set = list(client_set)
                            tmp_w_part = np.array(tmp_w_locals)[np.array(client_set)].tolist()
                            tmp_w_part = [tuple(elem) for elem in tmp_w_part]
                            tmp_w_full = tmp_w_part + [tmp_w_locals[idx]]

                            tmp_model_trainer = copy.deepcopy(self.model_trainer)
                            tmp_w_fake_global_full = self._aggregate(tmp_w_full)
                            tmp_model_trainer.set_model_params(tmp_w_fake_global_full)
                            tmp_m_full = self._valid_test_on_aggregator(
                                tmp_model_trainer.model, self.global_valid_data, self.device)

                            tmp_model_trainer = copy.deepcopy(self.model_trainer)
                            tmp_w_fake_global_part = self._aggregate(tmp_w_part)
                            tmp_model_trainer.set_model_params(tmp_w_fake_global_part)
                            tmp_m_part = self._valid_test_on_aggregator(
                                tmp_model_trainer.model, self.global_valid_data, self.device)
                            if isinstance(self.target_label, int):
                                if self.score == "F1":
                                    ap += tmp_m_full["F1"] - tmp_m_part["F1"]
                                elif self.score in ["Sensitivity", "Recall", "TPR", "tpr"]:
                                    ap += tmp_m_full["Recall"] - tmp_m_part["Recall"]
                                elif self.score in ["Precision", "PPV", "ppv"]:
                                    ap += tmp_m_full["Precision"] - tmp_m_part["Precision"]
                                else:
                                    ap += (tmp_m_full["test_correct"] - tmp_m_part["test_correct"]) / tmp_m_part["test_total"]
                            else:
                                ap += (tmp_m_full["test_correct"] - tmp_m_part["test_correct"]) / tmp_m_part["test_total"]

                    assert cnt != 0
                    client_idx = client_indexes[idx]
                    # sv[client_idx] += (ap / cnt)

                    tmp_w_single = self._aggregate(copy.deepcopy([tmp_w_locals[idx]]))
                    tmp_model_trainer = copy.deepcopy(self.model_trainer)
                    tmp_model_trainer.set_model_params(tmp_w_single)
                    tmp_m_single = self._valid_test_on_aggregator(
                        tmp_model_trainer.model, self.global_valid_data, self.device
                    )
                    tmp_client_accuracy = tmp_m_single["test_correct"] / tmp_m_single["test_total"]
                    client_dict[round_idx][client_idx] = tmp_client_accuracy
                    ap += tmp_client_accuracy
                    cnt += 1
                    sv[client_idx] = (ap / cnt)
                    # sv[client_idx] += (ap / cnt)
                    phi[client_idx] = alpha * phi[client_idx] + beta * sv[client_idx]
                    logging.info(f"Client {client_idx} sv={sv[client_idx]} phi={phi[client_idx]}")

                t_tail = time.time()
                t_cost = t_tail - t_head
                logging.info(f"[cost {t_cost}]")

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                res_dict[round_idx] = self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    res_dict[round_idx] = self._local_test_on_all_clients(round_idx)

            if round_idx not in res_dict.keys():
                res_dict[round_idx] = {}
            res_dict[round_idx]["Global/Acc"], res_dict[round_idx]["Global/Recall"] = self._validate_global_model(
                self.model_trainer.model, self.test_global, self.device)

            logging.info(f"Round {round_idx}"
                         f"\n\tGlobal/Recall = {res_dict[round_idx]['Global/Recall']}"
                         f"\n\tGlobal/Acc = {res_dict[round_idx]['Global/Acc']}")

            phi_dict[round_idx] = copy.deepcopy(phi)
            sv_dict[round_idx] = copy.deepcopy(sv)
            time_dict[round_idx] = t_cost
        joblib.dump(phi_dict, ".tmp_phi.pkl")
        joblib.dump(res_dict, ".tmp_res.pkl")
        joblib.dump(sv_dict, ".tmp_sv.pkl")
        joblib.dump(time_dict, ".tmp_time.pkl")
        joblib.dump(client_dict, ".tmp_client.pkl")

    def _validate_global_model(self, model: torch.nn.Module, data, device):
        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_total": 0
        }
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.to(device)
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(data):
                x = x.to(device)
                target = target.to(device)
                y_true += target.tolist()
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                y_pred += predicted.tolist()
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

            global_acc = metrics["test_correct"] / metrics["test_total"]
            if self.target_label is None:
                global_recall = None
            else:
                global_recall = recall_score(y_true, y_pred, average=None, labels=[self.target_label])[0]

        return global_acc, global_recall

    def _valid_test_on_aggregator(self, model: torch.nn.Module, data, device):
        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_total": 0
        }
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.to(device)
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(data):
                x = x.to(device)
                target = target.to(device)
                y_true += target.tolist()
                pred_res = model(x)
                # if isinstance(pred_res, tuple) and len(pred_res) == 2:
                #     pred = pred_res[1]
                # else:
                pred = pred_res
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                y_pred += predicted.tolist()
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        if isinstance(self.target_label, int):
            metrics["F1"] = f1_score(y_true, y_pred, average=None, labels=[self.target_label])[0]
            metrics["Recall"] = recall_score(y_true, y_pred, average=None, labels=[self.target_label])[0]
            metrics["Precision"] = precision_score(y_true, y_pred, average=None, labels=[self.target_label])[0]

        return metrics

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round, phi, sampling_filter=None):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            P = []
            for idx in range(self.args.client_num_in_total):
                if sampling_filter == "exp":
                    P.append(np.exp(phi[idx]))
                else:
                    P.append(1)
            _P = np.array(P) / (np.sum(P) + 1e-13)
            client_indexes = np.random.choice(
                range(client_num_in_total),
                size=num_clients, replace=False, p=_P
            ).tolist()

            # if len(set(phi)) == 1:
            #     np.random.seed((round_idx + 1) * self.seed)
            #     client_indexes = np.random.choice(
            #         range(client_num_in_total), num_clients, replace=False
            #     ).tolist()
            # else:
            #     sorted_indexes = np.argsort(phi, kind=random.choice([
            #         'quicksort', 'mergesort', 'heapsort', 'stable'
            #     ])).tolist()
            #     client_indexes = sorted_indexes[-num_clients:]
            #     if isinstance(ratio, float) and 0 < ratio < 1:
            #         partial_client_indexes = copy.deepcopy(client_indexes)
            #         random.seed(self.seed)
            #         lucky_cat = random.choice(partial_client_indexes)
            #         # lucky_cat = client_indexes[0]
            #         partial_client_indexes.remove(lucky_cat)
            #         lucky_dogs = sorted_indexes[:-num_clients] + [lucky_cat]
            #         avg_prob = [(1 - ratio) / (len(lucky_dogs) - 1)] * (len(lucky_dogs) - 1) + [ratio]
            #         lucky_dog = np.random.choice(lucky_dogs, replace=False, p=avg_prob)
            #         logging.info(f"Lucky dog: {lucky_dog}, Lucky cat: {lucky_cat}")
            #         client_indexes = [lucky_dog] + partial_client_indexes

        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(
            range(test_data_num), min(num_samples, test_data_num)
        )
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size
        )
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": [], "recall": [], "precision": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.calc_class_weight(self.train_data_local_dict[client_idx])
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(
                copy.deepcopy(train_local_metrics["test_total"])
            )
            train_metrics["num_correct"].append(
                copy.deepcopy(train_local_metrics["test_correct"])
            )
            train_metrics["losses"].append(
                copy.deepcopy(train_local_metrics["test_loss"])
            )

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(
                copy.deepcopy(test_local_metrics["test_total"])
            )
            test_metrics["num_correct"].append(
                copy.deepcopy(test_local_metrics["test_correct"])
            )
            test_metrics["losses"].append(
                copy.deepcopy(test_local_metrics["test_loss"])
            )
            test_metrics["recall"].append(
                copy.deepcopy(test_local_metrics["test_recall"])
            )
            test_metrics["precision"].append(
                copy.deepcopy(test_local_metrics["test_precision"])
            )

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(
            train_metrics["num_samples"]
        )
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)
        
        train_acc_list = np.array(train_metrics['num_correct']) / np.array(train_metrics['num_samples'])
        logging.info(f"Train/Acc: {train_acc_list.tolist()}")

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        test_acc_list = np.array(test_metrics['num_correct']) / np.array(test_metrics['num_samples'])
        logging.info(f"Test/Acc: {test_acc_list.tolist()}")

        return {
                "Test/Acc": test_acc_list.tolist(),
                "Train/Acc": train_acc_list.tolist(),
                "Test/Recall": test_metrics['recall'],
                "Test/Precision": test_metrics['precision']
        }

    def _local_test_on_validation_set(self, round_idx):

        logging.info(
            "################local_test_on_validation_set : {}".format(round_idx)
        )

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception(
                "Unknown format to log metrics for dataset {}!" % self.args.dataset
            )

        logging.info(stats)
