import argparse
import numpy as np

from FLF.TorchFederatedLearner import (
    TorchFederatedLearnerTechnicalConfig,
    ToLargeLearningRateExcpetion,
)
from FLF.TorchFederatedLearnerEMNIST import TorchFederatedLearnerEMNISTConfig

import common
from mutil.Empty import Empty

args_data = (
    ("server_lr", float, "Server learning rate"),
    ("client_lr", float, "Client learning rate"),
    ("server_opt", str, "Server optimizer"),
    ("client_opt", str, "Client optimizer"),
    ("--avg", None, "Average the state of the client optimizer"),
    ("n_clients", int, "Number of clients"),
    ("n_clients_per_round", int, "Number of clients sampled per round"),
    ("n_epochs", int, "Number of epoch per round per client"),
    ("n_rounds", int, "Number of rounds"),
    ("--eval_last_20", None, "Run the evaluation for the last 20 rounds and average them. If this is not set the evaluation runs for the last round."),
    ("--scaffold", None, "Use scaffold")
)

parser = argparse.ArgumentParser()
for n, t, h in args_data:
    if n[:2] == "--":
        parser.add_argument(n, help=h, action="store_true")
    else:
        parser.add_argument(n, type=t, help=h)
args = parser.parse_args()

server_lr = args.server_lr
client_lr = args.client_lr
server_opt = args.server_opt
client_opt = args.client_opt
client_opt_strategy = "avg" if args.avg else "reinit"

max_rounds = args.n_rounds
n_clients_per_round = args.n_clients_per_round
NC = args.n_clients
C = n_clients_per_round / NC
B = 20
is_iid = False
model = "CNN"
E = args.n_epochs
scaffold = args.scaffold

test_last = 20 if args.eval_last_20 else 1

config = TorchFederatedLearnerEMNISTConfig(
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=common.get_name(client_opt),
    CLIENT_OPT_ARGS=common.get_args(client_opt),
    CLIENT_OPT_STRATEGY=client_opt_strategy,
    SERVER_OPT=common.get_name(server_opt),
    SERVER_OPT_ARGS=common.get_args(server_opt),
    SERVER_LEARNING_RATE=server_lr,
    IS_IID_DATA=is_iid,
    BATCH_SIZE=B,
    CLIENT_FRACTION=C,
    N_CLIENTS=NC,
    N_EPOCH_PER_CLIENT=E,
    MAX_ROUNDS=max_rounds,
    MODEL=model,
    SCAFFOLD=scaffold
)
config_technical = TorchFederatedLearnerTechnicalConfig(
    BREAK_ROUND=300,
    EVAL_ROUND=1,
    TEST_LAST=20,
    STORE_OPT_ON_DISK=True,
    STORE_MODEL_IN_RAM=True,
)
name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
experiment = Empty()
try:
    common.do_training_emnist(experiment, name, config, config_technical)
except ToLargeLearningRateExcpetion:
    pass
