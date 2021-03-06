from comet_ml import Experiment
import numpy as np

from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig, ToLargeLearningRateExcpetion
from FLF.TorchFederatedLearnerEMNIST import TorchFederatedLearnerEMNISTConfig

import common


server_opt = "Yogi"
client_opt = "Yogi"
client_opt_strategy = "avg"

max_rounds = 100
n_clients_per_round = 10
NC = 3400
C = n_clients_per_round / NC
B = 20
is_iid = False
model = "CNN"

for _ in range(10):
    for _, m in common.get_besr_lrs_from_exps(server_opt, client_opt_strategy[0], client_opt).iterrows():
        E = m["E"]
        server_lr = 10 ** m["slr"]
        client_lr = 10 ** m["clr"]
        project_name = f"{model}{NC}c{E}e{max_rounds}r{n_clients_per_round}f-{server_opt}-{client_opt_strategy[0]}-{client_opt}-compare-best-lr"
        config = TorchFederatedLearnerEMNISTConfig(
            CLIENT_LEARNING_RATE=client_lr,
            CLIENT_OPT=common.get_name(client_opt),
            CLIENT_OPT_ARGS=common.get_args(client_opt),
            # CLIENT_OPT_L2=1e-4,
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
        )
        config_technical = TorchFederatedLearnerTechnicalConfig(
            BREAK_ROUND=300, EVAL_ROUND=10, TEST_LAST=20
        )
        name = f"{config.SERVER_OPT}: {config.SERVER_LEARNING_RATE} - {config.CLIENT_OPT_STRATEGY} - {config.CLIENT_OPT}: {config.CLIENT_LEARNING_RATE}"
        experiment = Experiment(
            workspace="federated-learning-scaffold", project_name=project_name
        )
        try:
            common.do_training_emnist(experiment, name, config, config_technical)
        except ToLargeLearningRateExcpetion:
            pass
