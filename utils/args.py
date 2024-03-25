import os
from omegaconf import OmegaConf
from core.da_model import DomainAdaptationGenerator
import inspect

base_generator_args = {k: v.default for k, v in inspect.signature(DomainAdaptationGenerator.__init__).parameters.items()
                       if
                       k != 'self'}

DEFAULT_CONFIG_DIR = 'configs'


def get_generator_args(generator_name, base_args, conf_args):
    return OmegaConf.create(
        {generator_name: OmegaConf.merge(base_args, conf_args)}
    )


def load_config():
    base_gen_args_config = OmegaConf.structured(base_generator_args)

    conf_cli = OmegaConf.from_cli()
    conf_cli.exp.config_dir = DEFAULT_CONFIG_DIR
    if not conf_cli.get('exp', False):
        raise ValueError("No config")

    config_path = os.path.join(DEFAULT_CONFIG_DIR, conf_cli.exp.config)
    conf_file = OmegaConf.load(config_path)

    conf_generator_args = conf_file.generator_args
    #
    generator_args = get_generator_args(
        conf_file.training.generator,
        base_gen_args_config,
        conf_generator_args
    )

    gen_args = OmegaConf.create({
        'generator_args': generator_args
    })

    config = OmegaConf.merge(conf_file, conf_cli)
    config = OmegaConf.merge(config, gen_args)
    return config
