import torch
import sys
import os
os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, '/eos/home-m/majorgen/particle_transformer')
from pquant import dst_config
from networks.ParticleTransformer_pquant import ParticleTransformer

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(config=config, **kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors / 2048.0, mask=mask)

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=False)
        for module in self.modules():
            if hasattr(module, 'final_compression_done'):
                fcd = module.final_compression_done
                if isinstance(fcd, torch.Tensor):
                    fcd.data.fill_(True)
                else:
                    module.final_compression_done = True
        return result

def get_model(data_config, **kwargs):
    config = dst_config()
    config.quantization_parameters.enable_quantization = True
    config.quantization_parameters.quantize_input = True
    config.quantization_parameters.quantize_output = False
    config.quantization_parameters.default_data_integer_bits = 4
    config.quantization_parameters.default_data_fractional_bits = 14
    config.quantization_parameters.default_data_keep_negatives = 1
    config.quantization_parameters.default_weight_integer_bits = 0
    config.quantization_parameters.default_weight_fractional_bits = 14
    config.quantization_parameters.default_weight_keep_negatives = 1
    config.quantization_parameters.overflow_mode_data = "SAT"
    config.quantization_parameters.overflow_mode_parameters = "SAT_SYM"
    config.quantization_parameters.round_mode = "RND"
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    model = ParticleTransformerWrapper(config, **cfg)
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }
    return model, model_info

def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()