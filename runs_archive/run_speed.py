from comet_ml import Experiment
from FLF.TorchFederatedLearnerCIFAR100 import TorchFederatedLearnerCIFAR100Config
from FLF.TorchFederatedLearner import TorchFederatedLearnerTechnicalConfig

import common


project_name = "serveropt-test"

max_rounds = 10 # 3000
C = 10 / 500
NC = 500
E = 1
B = 20
is_iid = False
server_lr = 1
client_lr = 0.0316
client_opt = "SGD"
client_opt_strategy = "reinit"
# image_norm = "tflike"
# TODO a paraméterek helytelen nevére nem adott hibát
for sopt in ["SGD", None]:
    config = TorchFederatedLearnerCIFAR100Config(
        BREAK_ROUND=1500,
        CLIENT_LEARNING_RATE=client_lr,
        CLIENT_OPT=client_opt,
        CLIENT_OPT_ARGS=common.get_args(client_opt),
        CLIENT_OPT_L2=1e-4,
        CLIENT_OPT_STRATEGY=client_opt_strategy,
        SERVER_OPT=sopt,
        # SERVER_OPT_ARGS=common.get_args(server_opt),
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
        SEED=42,
    )
    config_technical = TorchFederatedLearnerTechnicalConfig()
    name = f"{sopt}"
    experiment = Experiment(workspace="federated-learning", project_name=project_name)
    common.do_training(experiment, name, config, config_technical)
