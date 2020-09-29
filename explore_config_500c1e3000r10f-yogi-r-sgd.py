import comet_ml  # Comet.ml needs to be imported before PyTorch
import torch as th

from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig
from FLF.TorchFederatedLearnerCIFAR100 import (
    TorchFederatedLearnerCIFAR100,
    TorchFederatedLearnerCIFAR100Config,
)
from FLF.hyperopt.AdvancedGridLearningRate import explore_lr
import common


server_lr = 0.01
client_lr = 0.1
server_opt = "Yogi"
client_opt = "SGD"
client_opt_strategy = "reinit"

max_rounds = 3000
n_clients_per_round = 10
NC = 500
C = n_clients_per_round / NC
E = 1
B = 20
is_iid = False
project_name = f"{NC}c{E}e{max_rounds}r{n_clients_per_round}f-{server_opt}-{client_opt_strategy[0]}-{client_opt}"

config_technical = TorchFederatedLearnerTechnicalConfig(BREAK_ROUND=300)

config = TorchFederatedLearnerCIFAR100Config(
    CLIENT_LEARNING_RATE=client_lr,
    CLIENT_OPT=common.get_name(client_opt),
    CLIENT_OPT_ARGS=common.get_args(client_opt),
    CLIENT_OPT_L2=1e-4,
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
    IMAGE_NORM="recordwisefull",
    NORM="group",
    INIT="tffed",
    AUG="basicf",
)

explore_lr(
    project_name,
    TorchFederatedLearnerCIFAR100,
    config,
    config_technical,
    "federated-learning-hpopt",
    True
)