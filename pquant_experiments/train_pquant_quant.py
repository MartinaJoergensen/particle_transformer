import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ['WANDB_API_KEY'] = "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"
output_dir = None
import logging
import torch
from pquant import PQConfig, dst_config, train_model, apply_final_compression
from pquant.core.utils import write_config_to_yaml
from train_weaver_pquant import train_load, model_setup
from tools_pquant import train_classification, evaluate_classification, TensorboardHelper
import yaml
import wandb

logging.basicConfig(level=logging.INFO)

def train_particle_transformer(model, steps_per_epoch, train_loader, loss_func, dev, writer, epoch, optimizer, scheduler, *args, **kwargs):
    train_classification(model, loss_func, optimizer, scheduler, train_loader, dev, epoch, steps_per_epoch, None, writer)
def validate_particle_transformer(model, steps_per_epoch_val, test_loader, loss_func, dev, writer, epoch, *args, **kwargs):
    global output_dir
    acc = evaluate_classification(model, test_loader, dev, epoch, True, loss_func, steps_per_epoch_val, tb_helper=writer)
    logging.info(f"Epoch {epoch} — Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    from pquant import get_layer_keep_ratio
    ratio = get_layer_keep_ratio(model)
    logging.info(f"Epoch {epoch} — Remaining weights: {ratio:.4f} ({ratio*100:.2f}%)")
    scheduler = kwargs.get('scheduler')
    if wandb.run is not None:
        wandb.log({"eval/remaining_weights": ratio, "train/lr": scheduler.get_last_lr()[0] if scheduler else None, "epoch": epoch})
    if output_dir:
        torch.save(model.state_dict(), f"{output_dir}/checkpoint_latest.pt")
def get_model_data_loss(config: PQConfig, device: str):
    with open(f"configs/particle_transformer_quant.yaml", "r") as f:
        particle_transformer_config = yaml.safe_load(f)
    from types import SimpleNamespace
    particle_transformer_config = SimpleNamespace(**particle_transformer_config)
    train_loader, val_loader, data_config, _, _ = train_load(particle_transformer_config)
    model, _, loss_func = model_setup(config, particle_transformer_config, data_config, device=device)
    train_func = train_particle_transformer
    valid_func = validate_particle_transformer
    model = model.to(device)
    return model, train_loader, val_loader, loss_func, train_func, valid_func, data_config, particle_transformer_config
def main(config: PQConfig, weight_f_bits: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model, train_loader, val_loader, loss_func, train_func, valid_func,  data_config, particle_transformer_config = get_model_data_loss(
        config, device
    )
    tb = TensorboardHelper(tb_comment=particle_transformer_config.tensorboard, tb_custom_fn=particle_transformer_config.tensorboard_custom_fn)
    global output_dir
    output_dir = f"/eos/home-m/majorgen/particle_transformer/pquant_experiments/results/quant_scan_wf{weight_f_bits}"
    write_config_to_yaml(config, f"{output_dir}/config.yaml")
    parameters = list(model.named_parameters())
    threshold_params = [v for n, v in parameters if "threshold" in n and v.requires_grad]
    rest_params = [v for n, v in parameters if "threshold" not in n and v.requires_grad]
    optimizer = torch.optim.Adam(
            [
                {
                    "params": threshold_params,
                    "weight_decay": 0.,
                },
                {"params": rest_params, "weight_decay": 1e-4},
            ],
            lr=1e-3,
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[28,36,44], gamma=0.1
    )
    torch.save(model.state_dict(), f"{output_dir}/before_training.pt")
    trained_model = train_model(
        model=model,
        config=config,
        train_func=train_func,
        valid_func=valid_func,
        train_loader=train_loader,
        test_loader=val_loader,
        dev=device,
        loss_func=loss_func,
        writer=tb,
        optimizer=optimizer,
        scheduler=scheduler,
        steps_per_epoch = 50000,
        steps_per_epoch_val = 10000
    )
    torch.save(trained_model.state_dict(), f"{output_dir}/final_before_remove_model.pt")
    trained_model = apply_final_compression(trained_model)
    torch.save(trained_model.state_dict(), f"{output_dir}/final_model.pt")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-f-bits", type=int, default=14)
    args = parser.parse_args()
    config = dst_config()
    config.training_parameters.epochs = 50
    config.pruning_parameters.enable_pruning = True
    config.pruning_parameters.alpha = 1e-9
    # Quantization — NEW
    config.quantization_parameters.enable_quantization = True
    config.quantization_parameters.quantize_input = True
    config.quantization_parameters.quantize_output = False
    config.quantization_parameters.default_data_integer_bits = 4
    config.quantization_parameters.default_data_fractional_bits = 11
    config.quantization_parameters.default_data_keep_negatives = 1
    config.quantization_parameters.default_weight_integer_bits = 0
    config.quantization_parameters.default_weight_fractional_bits = args.weight_f_bits
    config.quantization_parameters.default_weight_keep_negatives = 1
    config.quantization_parameters.overflow_mode_data = "SAT"
    config.quantization_parameters.overflow_mode_parameters = "SAT_SYM"
    config.quantization_parameters.round_mode = "RND"

    

    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="martina-jorgensen-cern",
        # Set the wandb project where this run will be logged.
        project="par-t-quant",
        name=f"scan-wf{args.weight_f_bits}",
        sync_tensorboard=True,
        # Track hyperparameters and run metadata.
        config=config.model_dump()
    )
    main(config,args.weight_f_bits)