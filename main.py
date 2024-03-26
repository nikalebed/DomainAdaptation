from pprint import pprint
from omegaconf import OmegaConf
from utils.args import load_config
from trainer import DomainAdaptationTrainer

config = load_config()
pprint(OmegaConf.to_container(config))

trainer = DomainAdaptationTrainer(config)
trainer.setup()
trainer.train()