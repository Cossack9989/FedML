from .mpi_p2p_mp.base_framework.algorithm_api import FedML_Base_distributed
from .mpi_p2p_mp.decentralized_framework.algorithm_api import (
    FedML_Decentralized_Demo_distributed,
)
from .mpi_p2p_mp.fedavg.FedAvgAPI import FedML_FedAvg_distributed
from .mpi_p2p_mp.fedavg_robust.FedAvgRobustAPI import FedML_FedAvgRobust_distributed

from .single_process.fedavg import FedAvgAPI
from .single_process.s_fedavg import S_FedAvgAPI
from .single_process.hs_fedavg import HS_FedAvgAPI
from ..constants import (
    FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK,
    FedML_FEDERATED_OPTIMIZER_FEDAVG,
    FedML_FEDERATED_OPTIMIZER_FEDOPT,
    FedML_FEDERATED_OPTIMIZER_FEDPROX,
    FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL,
    FedML_FEDERATED_OPTIMIZER_SPLIT_NN,
    FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL,
    FedML_FEDERATED_OPTIMIZER_FEDGAN,
    FedML_FEDERATED_OPTIMIZER_FEDAVG_ROBUST,
    FedML_FEDERATED_OPTIMIZER_FEDGKT,
    FedML_FEDERATED_OPTIMIZER_FEDNAS,
    FedML_FEDERATED_OPTIMIZER_FEDSEG,
    FedML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE,
)


class SimulatorSingleProcess:
    def __init__(self, args, device, dataset, model):
        if args.federated_optimizer == "FedAvg":
            self.fl_trainer = FedAvgAPI(args, device, dataset, model)
        elif args.federated_optimizer == "S-FedAvg":
            self.fl_trainer = S_FedAvgAPI(args, device, dataset, model)
        elif args.federated_optimizer == "HS-FedAvg":
            self.fl_trainer = HS_FedAvgAPI(args, device, dataset, model)
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()


class SimulatorMPI:
    def __init__(self, args, device, dataset, model, model_trainer=None):
        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            self.simulator = FedML_FedAvg_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK:
            self.simulator = FedML_Base_distributed(
                args, args.process_id, args.worker_num, args.comm
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDOPT:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDPROX:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_SPLIT_NN:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL:
            self.simulator = FedML_Decentralized_Demo_distributed(
                args, args.process_id, args.worker_num, args.comm
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDGAN:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG_ROBUST:
            self.simulator = FedML_FedAvgRobust_distributed(
                args,
                args.process_id,
                args.worker_num,
                device,
                args.comm,
                model,
                dataset,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDGKT:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDNAS:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDSEG:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE:
            pass
        else:
            raise Exception("Exception")

    def run(self):
        self.simulator.train()


class SimulatorNCCL:
    def __init__(self, args, device, dataset, model, model_trainer=None):
        if args.federated_optimizer == "FedAvg":
            self.simulator = None
        else:
            raise Exception("Exception")

    def run(self):
        self.simulator.train()
