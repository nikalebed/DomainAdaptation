from pprint import pprint
from omegaconf import OmegaConf
from utils.args import load_config
from trainer import DomainAdaptationTrainer
import random
import torch
import numpy as np

config = load_config()
pprint(OmegaConf.to_container(config))

seed = config.exp.seed

random.seed(seed)
torch.random.manual_seed(seed)
np.random.seed(seed)

trainer = DomainAdaptationTrainer(config)
trainer.setup()
trainer.train()
