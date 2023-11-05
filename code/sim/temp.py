from chanakya.RL_gpu import MultiOutputMAB
from chanakya.rl_configs import *

rl_model = MultiOutputMAB(mab_configs["frcnn_ucb"], device="cuda:3")
rl_model.load(".", "frcnn_ucb_argoverse_temp")
print(rl_model.agents[0].history)