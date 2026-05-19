import argparse

import torch
import yaml

from train_pquant_quant import get_model_data_loss, validate_particle_transformer
from pquant import get_layer_keep_ratio
from pquant.layers import PQDense

def validate(config, device, args):
    model, _, val_loader, loss_func, _, _, _, _ = get_model_data_loss(config, device)
    model.to(device)

    model.load_state_dict(torch.load(f"{args.config}/{args.checkpoint}"), strict=False)
    print("REMAINING WEIGHTS", get_layer_keep_ratio(model))
    validate_particle_transformer(model, 100, val_loader, loss_func, device, writer=None, epoch=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to PQConfig YAML file")
    parser.add_argument("--checkpoint", type=str, default="final_model.pt", help="Checkpoint filename inside --config dir")
    args = parser.parse_args()
    with open(args.config + "/config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    print(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate(cfg, device, args)