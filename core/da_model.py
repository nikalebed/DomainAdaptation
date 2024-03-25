import torch
from torch import nn
from gan_models.parametrized_model import ParametrizedGenerator, \
    ParametrizedModulatedConv2d


def requires_grad(model, flag=True):
    if isinstance(model, nn.Parameter):
        model.requires_grad = flag
        return

    for p in model.parameters():
        p.requires_grad = flag


class DomainAdaptationGenerator(nn.Module):
    def __init__(self, img_size=1024, latent_size=512, map_layers=8,
                 channel_multiplier=2, device='cuda', checkpoint_path=None):
        super().__init__()

        self.generator = ParametrizedGenerator(
            img_size, latent_size, map_layers,
            channel_multiplier=channel_multiplier
        ).to(device)

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.generator.load_state_dict(checkpoint["g_ema"], strict=False)

        self.generator.eval()

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def add_patches(self):
        self.generator.conv1.conv = ParametrizedModulatedConv2d(
            self.generator.conv1.conv)

        for conv_layer_ix in range(len(self.generator.convs)):
            self.generator.convs[
                conv_layer_ix].conv = ParametrizedModulatedConv2d(
                self.generator.convs[conv_layer_ix].conv
            )

    def forward(self,
                styles,
                params=None,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                noise=None,
                randomize_noise=True):
        return self.generator(styles,
                              params=params,
                              return_latents=return_latents,
                              truncation=truncation,
                              truncation_latent=self.mean_latent,
                              noise=noise,
                              randomize_noise=randomize_noise,
                              input_is_latent=input_is_latent)

    def freeze_layers(self, layer_list=None):
        """
        Disable training for all layers in list.
        """
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        """
        Enable training for all layers in list.
        """
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def get_all_layers(self):
        return list(self.generator.children())
