import os
import wandb
import torch
from core.da_model import DomainAdaptationGenerator
from core.parametrization import Parametrization
from utils.sg2_utils import get_stylegan_conv_dimensions, mixing_noise
from core.loss import ComposedLoss
import typing as tp
from core.inverters import BaseInverter, II2SInverter, e4eInverter
from core.batch_generators import DiFABaseClipBatchGenerator
from pathlib import Path
from utils.image_utils import t2im, construct_paper_image_grid, get_image_t, resize_batch
from tqdm import tqdm


class DomainAdaptationTrainer:

    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.current_step = None
        self.zs_for_logging = None

        self.source_generator = None
        self.trainable = None
        self.criterion = None
        self.optimizer = None

        self.style_image_full_res = None
        self.style_image_resized = None
        self.style_image_latent = None
        self.style_image_inverted_A = None

        self.clip_batch_generator = None
        self.image_inverter = BaseInverter()

    def setup(self):
        self.setup_source_generator()

        self.initial_logging()

        self.setup_trainable()
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_image_inverter()

        self.setup_style_image()
        self.log_target_images()

        self.setup_clip_batch_generator()

    def setup_source_generator(self):
        self.source_generator = DomainAdaptationGenerator(
            **self.config.generator_args[self.config.training.generator])
        self.source_generator.add_patches()  # TODO add options
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)

    def setup_optimizer(self):
        if self.config.training.da_type == "original":
            g_reg_every = self.config.optimization_setup.g_reg_every
            lr = self.config.optimization_setup.optimizer.lr

            g_reg_ratio = g_reg_every / (g_reg_every + 1)
            betas = self.config.optimization_setup.optimizer.betas

            self.optimizer = torch.optim.Adam(
                self.trainable.parameters(),
                lr=lr * g_reg_ratio,
                betas=(betas[0] ** g_reg_ratio, betas[1] ** g_reg_ratio),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.trainable.parameters(), **self.config.optimization_setup.optimizer
            )

    def setup_trainable(self):
        if self.config.training.da_type == 'original':
            self.trainable = DomainAdaptationGenerator(
                **self.config.generator_args[self.config.training.generator])
            self.trainable.add_patches()
            # trainable_layers = list(self.trainable.get_training_layers(
            #     phase=self.config.training.phase
            # ))
            # self.trainable.freeze_layers()
            # self.trainable.unfreeze_layers(trainable_layers)
        else:
            self.trainable = Parametrization(
                get_stylegan_conv_dimensions(self.source_generator.generator.size))

    def setup_criterion(self):
        # self.has_clip_loss = len(self.loss_function.clip.funcs) > 0
        self.criterion = ComposedLoss(self.config.optimization_setup)
        self.criterion.scc_loss.iter = self.config.training.iter_num

    def setup_style_image(self):
        style_image_t = get_image_t(self.config.training.target_class, self.source_generator.generator.size)

        self.style_image_latent = self.image_inverter.get_latents(
            style_image_t).detach().clone()
        self.style_image_resized = resize_batch(style_image_t, 256)
        self.style_image_inverted_A = self.forward_source(
            [self.style_image_latent], input_is_latent=True)

    def log_target_images(self):
        style_image_resized = t2im(self.style_image_resized.squeeze())
        st_im_inverted_A = t2im(self.style_image_inverted_A.squeeze())

        wandb.log({f"style_image/orig": wandb.Image(style_image_resized, caption=f"reference")})
        wandb.log({f"style_image/projected_A": wandb.Image(st_im_inverted_A, caption=f"projected reference")})

    @torch.no_grad()
    def initial_logging(self):
        self.zs_for_logging = [
            mixing_noise(16, 512, 0, self.config.training.device)
            for _ in range(self.config.logging.num_grid_outputs)
        ]

        wandb.init(project=self.config.exp.project,
                   name=self.config.exp.name,
                   dir=self.config.exp.root,
                   tags=tuple(self.config.exp.tags) if self.config.exp.tags else None,
                   notes=self.config.exp.notes,
                   config=dict(self.config))

        for idx, z in enumerate(self.zs_for_logging):
            images = self.forward_source(z)
            wandb.log(
                {f"src_domain_grids/{idx}": wandb.Image(construct_paper_image_grid(images), caption=f"originals")})

    @torch.no_grad()
    def forward_source(self, latents, **kwargs) -> torch.Tensor:
        sampled_images, _ = self.source_generator(latents, **kwargs)
        return sampled_images.detach()

    def forward_trainable(self, latents, **kwargs) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        if self.config.training.da_type == "original":
            sampled_images, _ = self.trainable(
                latents, **kwargs
            )
            params = None
        else:
            params = self.trainable()
            sampled_images, _ = self.source_generator(
                latents, params=params, **kwargs
            )

        return sampled_images, params

    def encode_batch(self, sample_z):
        frozen_img = self.forward_source(sample_z)
        trainable_img, params = self.forward_trainable(sample_z)

        clip_data = self.clip_batch_generator.calc_batch(frozen_img, trainable_img)
        inv_data = {
            'src_latents': self.image_inverter.get_latents(frozen_img),
            'trg_latents': self.image_inverter.get_latents(trainable_img),
            'iters': self.current_step
        }
        return {
            'clip_data': clip_data,
            'offsets': params,
            'inv_data': inv_data
        }

    def train_iter(self):
        z = mixing_noise(self.config.training.batch_size,
                         512,
                         self.config.training.mixing_noise,
                         self.config.training.device)
        batch = self.encode_batch(z)
        loss = self.criterion(batch)
        self.optimizer.zero_grad()
        loss['final'].backward()
        self.optimizer.step()
        return loss

    def all_to_device(self, device):
        self.source_generator.to(device)
        self.trainable.to(device)
        self.criterion.to(device)

    def train(self):
        self.all_to_device(self.device)
        for self.current_step in tqdm(range(1, self.config.training.iter_num + 1)):
            loss = self.train_iter()
            if self.current_step % self.config.logging.log_images == 0:
                loss.update(self.get_logger_images())
                wandb.log(loss)
        self.save_checkpoint()
        wandb.finish()

    def setup_clip_batch_generator(self):
        self.clip_batch_generator = DiFABaseClipBatchGenerator(self.config)

    def setup_image_inverter(self):
        if self.config.inversion.method == 'e4e':
            self.image_inverter = e4eInverter()
        elif self.config.inversion.method == 'ii2s':
            self.image_inverter = II2SInverter()

    def get_checkpoint(self):
        state_dict = {
            "step": self.current_step,
            "trainable": self.trainable.state_dict(),
            "trainable_optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        return state_dict

    def get_checkpoint_name(self):
        base = f'{self.config.training.da_type}_{Path(self.config.training.target_class).stem}_{self.current_step}_checkpoint'
        base = os.path.join(self.config.exp.checkpoint_dir, base)
        filename = base + '{}.pt'
        counter = 0
        while os.path.isfile(filename.format(counter)):
            counter += 1
        filename = filename.format(counter)
        return filename

    def save_checkpoint(self):
        # if not self.config.checkpointing.is_on:
        #     return
        ckpt = self.get_checkpoint()
        if not os.path.exists(self.config.exp.checkpoint_dir):
            # Create a new directory because it does not exist
            os.makedirs(self.config.exp.checkpoint_dir)
        torch.save(ckpt, self.get_checkpoint_name())

    @torch.no_grad()
    def get_logger_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,
            })

        rec_img, _ = self.forward_trainable(
            [self.style_image_latent],
            input_is_latent=True,
        )

        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})

        dict_to_log = {k: wandb.Image(v, caption=f"iter = {self.current_step}") for k, v in dict_to_log.items()}
        return dict_to_log
