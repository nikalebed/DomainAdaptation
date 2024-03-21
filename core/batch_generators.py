from utils.common import load_clip
import torch.nn.functional as F


class BaseClipBatchGenerator:
    def __init__(self, config):
        self.config = config
        self.batch_generators = {}
        for visual_encoder in self.config.optimization_setup.visual_encoders:
            self.batch_generators[visual_encoder] = (
                load_clip(visual_encoder, device=self.config.training.device)
            )

    def setup_clip_encoders(self):
        self.clip_encoders = {}
        for visual_encoder in self.config.optimization_setup.visual_encoders:
            self.clip_encoders[visual_encoder] = (
                load_clip(visual_encoder, device=self.config.training.device)
            )

    def clip_encode_image(self, model, image, preprocess, norm=True):
        image_features = model.encode_image(preprocess(image))

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def calc_batch(self, frozen_img, trainable_img):
        raise NotImplementedError()


class DiFABaseClipBatchGenerator(BaseClipBatchGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.setup_hook = False
        self.hook_handlers = []
        self.hook_cache = {}

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
                output = F.layer_norm(output, visual_model.ln_post.normalized_shape, \
                                      visual_model.ln_post.weight.type(output.dtype), \
                                      visual_model.ln_post.bias.type(output.dtype), \
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

        # frozen_img = self.forward_source(sample_z)
        # trainable_img, offsets = self.forward_trainable(sample_z)

        if self.has_clip_loss:
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

                # trg_trainable_emb = self.clip_encode_image(model, style_image_inverted_B, preprocess)
                clip_data[visual_encoder_key].update({
                    'trg_encoded': trg_encoded,
                    'src_encoded': src_encoded,
                    'trg_domain_emb': self.trg_embeddings[visual_encoder_key],
                    'src_domain_emb': self.src_embeddings[visual_encoder_key],
                    # 'trg_trainable_emb': trg_trainable_emb,
                    # 'trg_emb': self.trg_embeddings[visual_encoder_key]
                })

            self._flush_hook_data()

        return clip_data
