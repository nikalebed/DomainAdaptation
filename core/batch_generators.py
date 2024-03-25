import torch
import numpy as np
from utils.common import load_clip
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms


class BaseClipBatchGenerator:
    def __init__(self, config):
        self.config = config
        self.device = self.config.training.device
        self._setup_clip_encoders()

    def _setup_clip_encoders(self):
        self.batch_generators = {}
        for visual_encoder in self.config.optimization_setup.visual_encoders:
            self.batch_generators[visual_encoder] = (
                load_clip(visual_encoder, device=self.device)
            )

    def clip_encode_image(self, model, image, preprocess, norm=True):
        image_features = model.encode_image(preprocess(image))

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def calc_batch(self, frozen_img, trainable_img):
        raise NotImplementedError()

    def _process_source_embeddings(self):
        self.src_embeddings = {}

        for clip_key, (m, p) in self.batch_generators.items():

            if self.config.emb.type == 'mean':
                src_dir = Path(self.config.emb.src_emb_dir)
                src_domain = Path(self.config.generator_args['checkpoint_path']).stem.split('-')[1]
                src_emb_name = f"{src_domain}_{clip_key.split('/')[1]}.pkl"
                src_emb_path = src_dir / src_emb_name

                if src_emb_path.exists():
                    import pickle as pkl
                    with open(str(src_emb_path), 'rb') as f:
                        X = pkl.load(f)
                        X = np.array(X)
                    mean = torch.from_numpy(np.mean(X, axis=0)).float().to(self.device)
                    mean /= mean.clone().norm(dim=-1, keepdim=True)
                    self.src_embeddings[clip_key] = mean
                else:
                    raise ValueError(f'no mean embedding of Source domain in dir: {src_emb_path}')

            # elif self.config.emb.type == 'online':
            #     self.src_embeddings[clip_key] = self._mean_clip_embedding(m, p, num=self.config.emb.num).unsqueeze(0)
            # elif self.config.emb.type == 'projected_target':
            #     self.src_embeddings[clip_key] = self.clip_encode_image(m, self.style_image_inverted_A, p)
            # elif self.config.emb.type == 'text':
            #     ...
            else:
                raise ValueError('Unknown emb type')


class DiFABaseClipBatchGenerator(BaseClipBatchGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.setup_hook = False
        self.hook_handlers = []
        self.hook_cache = {}
        self._setup_hooks()
        self._setup_embedding()

    def _setup_embedding(self):
        self._process_target_embeddings()
        self._process_source_embeddings()

    @torch.no_grad()
    def _process_target_embeddings(self):
        self.trg_embeddings = {}
        self.trg_keys = {}
        self.trg_tokens = {}

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        from PIL import Image
        tr = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        for clip_key, (m, p) in self.batch_generators.items():
            im = tr(Image.open(self.config.training.target_class)).unsqueeze(0).to(self.device)

            encoding = m.encode_image(im)
            encoding /= encoding.clone().norm(dim=-1, keepdim=True)

            self.trg_embeddings[clip_key] = encoding

            if self.setup_hook:
                trg_tokens = self.hook_cache[clip_key]['feat_tokens'][0]
                # trg_tokens /= trg_tokens.clone().norm(dim=-1, keepdim=True)

                trg_keys = self.hook_cache[clip_key]['feat_keys'][0]
                self.trg_keys[clip_key] = trg_keys
                self.trg_tokens[clip_key] = trg_tokens

            self._flush_hook_data()

    def _setup_hooks(self):
        if 'difa_local' in self.config.optimization_setup.loss_funcs:
            self.setup_hook = True

        if self.setup_hook:
            self.hook_cache = {k: {
                'feat_keys': [],
                'feat_tokens': [],
                'gen_attn_weights': [],
            } for k in self.batch_generators}
            self.hook_handlers = []

            self._register_hooks(layer_ids=[self.config.training.clip_layer], facet='key')

    def _get_hook(self, clip_key, facet):
        visual_model = self.batch_generators[clip_key][0]
        if facet in ['token']:
            def _hook(model, input, output):
                input = model.ln_1(input[0])
                attnmap = model.attn(input, input, input, need_weights=True, attn_mask=model.attn_mask)[1]
                self.hook_cache[clip_key]['feat_tokens'].append(output[1:].permute(1, 0, 2))
                self.hook_cache[clip_key]['gen_attn_weights'].append(attnmap)

            return _hook
        elif facet == 'feat':
            def _outer_hook(model, input, output):
                output = output[1:].permute(1, 0, 2)  # LxBxD -> BxLxD
                # TODO: Remember to add VisualTransformer ln_post, i.e. LayerNorm
                output = F.layer_norm(output, visual_model.ln_post.normalized_shape,
                                      visual_model.ln_post.weight.type(output.dtype),
                                      visual_model.ln_post.bias.type(output.dtype),
                                      visual_model.ln_post.eps)
                output = output @ visual_model.proj
                self.hook_cache[clip_key]['feat_tokens'].append(output)

            return _outer_hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            N, B, C = input.shape
            weight = module.in_proj_weight.detach()
            bias = module.in_proj_bias.detach()
            qkv = F.linear(input, weight, bias)[1:]  # remove cls key
            qkv = qkv.reshape(-1, B, 3, C).permute(2, 1, 0, 3)  # BxNxC
            self.hook_cache[clip_key]['feat_keys'].append(qkv[facet_idx])

        return _inner_hook

    def _register_hooks(self, layer_ids, facet='key'):
        for clip_name, (model, preprocess) in self.batch_generators.items():
            for block_idx, block in enumerate(model.visual.transformer.resblocks):
                if block_idx in layer_ids:
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(clip_name, 'token')))
                    assert facet in ['key', 'query', 'value']
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(clip_name, facet)))

    def _unregister_hooks(self):
        for handle in self.hook_handlers:
            handle.remove()

        self.hook_handlers = []

    def _flush_hook_data(self):
        if self.setup_hook:
            self.hook_cache = {k: {
                'feat_keys': [],
                'feat_tokens': [],
                'gen_attn_weights': [],
            } for k in self.batch_generators}

    def calc_batch(self, frozen_img, trainable_img):
        self._flush_hook_data()
        clip_data = {k: {} for k in self.batch_generators}

        for visual_encoder_key, (model, preprocess) in self.batch_generators.items():

            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)
            trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)

            if self.setup_hook:
                src_tokens = self.hook_cache[visual_encoder_key]['feat_tokens'][0]
                src_tokens /= src_tokens.clone().norm(dim=-1, keepdim=True)

                trg_tokens = self.hook_cache[visual_encoder_key]['feat_tokens'][1]
                trg_tokens /= trg_tokens.clone().norm(dim=-1, keepdim=True)

                clip_data[visual_encoder_key].update({
                    'trg_tokens': trg_tokens,
                    'src_tokens': src_tokens,
                    'trg_tokens_style': self.trg_tokens[visual_encoder_key]
                })

            clip_data[visual_encoder_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': self.trg_embeddings[visual_encoder_key],
                'src_domain_emb': self.src_embeddings[visual_encoder_key],
            })

        self._flush_hook_data()

        return clip_data
