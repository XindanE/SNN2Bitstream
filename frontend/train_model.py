# This file is part of SNN2Bitstream.
# Copyright (C) 2026 Xindan Zhang, Sorbonne Université, CNRS, LIP6

# SNN2Bitstream is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SNN2Bitstream is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import importlib
import toml
import torch
import torch.optim as optim
import snntorch.functional as SF
from snntorch import spikegen
import argparse
import copy

from frontend.qat import (convert_to_qat_model, extract_qat_scales,
                      convert_from_qat_model, fold_bn_into_model)
from frontend.models.fcn_model import FCNNet
from frontend.models.csnn_model import CSNNNet
from frontend.standardize_model import standardize_state_dict as _standardize_sd
from frontend.datasets import (default_encoding as _ds_default_encoding,
                               module_name as _ds_module_name)


def get_loaders(dataset_cfg, batch_size, timesteps, encoding="repeat"):
    dataset_name = _ds_module_name(dataset_cfg["kind"])
    try:
        dataset_module = importlib.import_module(f"frontend.datasets.{dataset_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Dataset {dataset_cfg['kind']} not supported.")

    static_datasets = {"mnist"}  # no timesteps parameter
    temporal_datasets = {"nmnist", "dvsgesture", "cifar10dvs"}  # need timesteps parameter

    if dataset_name in static_datasets:
        return dataset_module.get_loaders(dataset_cfg, batch_size, encoding=encoding)
    elif dataset_name in temporal_datasets:
        return dataset_module.get_loaders(dataset_cfg, batch_size, timesteps)
    else:
        raise ValueError(f"Dataset {dataset_cfg['kind']} not supported.")


def encode_input_fcn(data, timesteps, encoding="repeat"):
    """Encode input for FCN forward pass.

    Static images [B,C,H,W] -> [T,B,D] (flattened and repeated/rate-encoded).
    Temporal data [B,T,C,H,W] -> [T,B,C,H,W] (permute only; FCNNet.forward flattens internally).
    """
    if data.dim() == 5:
        return data.permute(1, 0, 2, 3, 4)  # [B,T,C,H,W] -> [T,B,C,H,W]
    data_flat = data.view(data.size(0), -1)  # [B,D]
    if encoding == "rate":
        return spikegen.rate(data_flat, num_steps=timesteps)  # [T,B,D]
    else:
        return data_flat.unsqueeze(0).repeat(timesteps, 1, 1)  # [T,B,D]


def encode_input_csnn(data, timesteps, encoding="repeat"):
    """Encode static image data for CSNN: [B,C,H,W] -> [B,T,C,H,W]"""
    if data.dim() == 5:
        return data  # temporal dataset, already [B,T,C,H,W]
    if encoding == "rate":
        # Flatten, rate-encode, reshape back.
        B, C, H, W = data.shape
        data_flat = data.view(B, -1)  # [B, C*H*W]
        spikes = spikegen.rate(data_flat, num_steps=timesteps)  # [T,B,C*H*W]
        return spikes.view(timesteps, B, C, H, W).permute(1, 0, 2, 3, 4)  # [B,T,C,H,W]
    else:
        return data.unsqueeze(1).repeat(1, timesteps, 1, 1, 1)  # [B,T,C,H,W]


def build_model(model_cfg, input_dim, output_dim):
    kind = model_cfg["kind"].lower()
    params = model_cfg["params"]

    model_class = {"fcn": FCNNet, "csnn": CSNNNet}.get(kind)
    if model_class is None:
        raise ValueError(f"Unsupported model kind: {model_cfg['kind']}")

    # Filter out non-model params before passing to constructor
    model_params = {k: v for k, v in params.items() if k != "encoding"}
    return model_class(input_dim=input_dim, output_dim=output_dim, **model_params)


def profile_input_data(loader, max_batches=50):
    """Profile input data statistics for auto-adaptive fixed-point type selection.

    Scans the first max_batches batches to determine {min, max, is_integer, is_signed}.
    """
    global_min, global_max = float('inf'), float('-inf')
    all_integer = True
    for batch_idx, (data, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        flat = data.reshape(data.size(0), -1)
        global_min = min(global_min, flat.min().item())
        global_max = max(global_max, flat.max().item())
        if all_integer and torch.abs(flat - torch.round(flat)).max().item() > 1e-6:
            all_integer = False
    return {
        "min": float(global_min),
        "max": float(global_max),
        "is_integer": bool(all_integer),
        "is_signed": bool(global_min < 0),
    }


def detect_binary_input(loader, kind, timesteps, encoding):

    try:
        data, _ = next(iter(loader))
    except StopIteration:
        return False
    x = encode_input_fcn(data, timesteps, encoding) if kind == "fcn" \
        else encode_input_csnn(data, timesteps, encoding)
    u = torch.unique(x)
    return bool(u.numel() <= 2 and torch.all((u == 0) | (u == 1)).item())


def _encode_and_forward(model, kind, data, timesteps, encoding):

    if kind == "fcn":
        x = encode_input_fcn(data, timesteps, encoding)
    else:
        x = encode_input_csnn(data, timesteps, encoding)
    result = model(x)
    if isinstance(result, tuple):
        spikes, mems = result
        if kind != "fcn":
            spikes = spikes.permute(1, 0, 2)  # [B,T,C] -> [T,B,C]
            mems   = mems.permute(1, 0, 2)
        return spikes, mems
    spikes = result
    if kind != "fcn":
        spikes = spikes.permute(1, 0, 2)
    return spikes


def evaluate(model, kind, loader, device, timesteps, encoding="repeat", loss_type="ce_count"):
    """Evaluate model accuracy on a data loader."""
    import torch.nn.functional as F_nn
    use_mem_loss = (loss_type == "mem_ce")
    if use_mem_loss:
        criterion = None
    elif loss_type == "mse_count":
        criterion = SF.mse_count_loss()
    else:
        criterion = SF.ce_count_loss()

    prev_output_mem = getattr(model, 'output_mem', False)
    if use_mem_loss:
        model.output_mem = True

    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            result = _encode_and_forward(model, kind, data, timesteps, encoding)
            if use_mem_loss:
                spikes, mems = result  # [T,B,C] each
                T = mems.size(0)
                loss = sum(F_nn.cross_entropy(mems[t], targets) for t in range(T))
            else:
                spikes = result
                loss = criterion(spikes, targets)
            total_loss += loss.item()
            _, pred = spikes.sum(0).max(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

    model.output_mem = prev_output_mem
    model.train()
    return total_loss / len(loader), 100.0 * correct / total


def train_with_history(model, kind, loader, optimizer, device,
                       epochs, timesteps, test_loader=None,
                       encoding="repeat", grad_clip=0.0, scheduler=None,
                       scheduler_per_epoch=False, loss_type="ce_count",
                       label_smoothing=0.0):
    """Train model and return per-epoch history and best model weights."""
    import torch.nn.functional as F_nn
    use_mem_loss = (loss_type == "mem_ce")
    if use_mem_loss:
        model.output_mem = True
        criterion = None
        print("[Info] Using membrane potential CE loss (per-timestep)")
    elif loss_type == "mse_count":
        criterion = SF.mse_count_loss()
        print("[Info] Using MSE count loss")
    else:
        criterion = SF.ce_count_loss()
    history = []
    best_test_acc = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for batch_idx, (data, targets) in enumerate(loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            result = _encode_and_forward(model, kind, data, timesteps, encoding)
            if use_mem_loss:
                spikes, mems = result
                T = mems.size(0)
                loss = sum(F_nn.cross_entropy(mems[t], targets,
                                              label_smoothing=label_smoothing)
                           for t in range(T))
            else:
                spikes = result
                loss = criterion(spikes, targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None and not scheduler_per_epoch:
                scheduler.step()
            total_loss += loss.item()
            with torch.no_grad():
                _, pred = spikes.sum(0).max(1)
                correct += (pred == targets).sum().item()
            total += targets.size(0)
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss {avg_loss:.4f} | Acc {acc:.2f}%", end="\r")

        train_loss = total_loss / len(loader)
        train_acc = 100.0 * correct / total

        test_loss, test_acc = 0.0, 0.0
        if test_loader is not None:
            test_loss, test_acc = evaluate(model, kind, test_loader, device, timesteps,
                                           encoding, loss_type=loss_type)
            print(f"\nEpoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                  f"Test Loss {test_loss:.4f} Acc {test_acc:.2f}%")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                print(f"  => New best model saved (test_acc={test_acc:.2f}%)")
        else:
            print(f"\nEpoch {epoch+1}/{epochs} Finished | "
                  f"Avg Loss {train_loss:.4f} | Acc {train_acc:.2f}%")

        if scheduler is not None and scheduler_per_epoch:
            scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

    return history, best_state_dict


def _build_optimizer(model, config, lr):
    """Build optimizer from [optimizer] config section. Returns (optimizer, weight_decay)."""
    weight_decay = config["optimizer"].get("weight_decay", 0.0)
    opt_kind = config["optimizer"].get("kind", "Adam").lower()
    if opt_kind == "sgd":
        momentum = config["optimizer"].get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        print(f"[Info] Optimizer: SGD (lr={lr}, momentum={momentum}, wd={weight_decay})")
    elif opt_kind == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[Info] Optimizer: AdamW (lr={lr}, wd={weight_decay})")
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[Info] Optimizer: Adam (lr={lr}, wd={weight_decay})")
    return optimizer, weight_decay


def _build_scheduler(optimizer, scheduler_kind, lr, epochs, steps_per_epoch, eta_min):
    """Build LR scheduler. Returns None if scheduler_kind is 'none'."""
    if scheduler_kind == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
        )
        print(f"[Info] Scheduler: OneCycleLR (max_lr={lr}, steps={steps_per_epoch * epochs})")
    elif scheduler_kind == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
        print(f"[Info] Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min={eta_min})")
    else:
        scheduler = None
    return scheduler


def main(config_path, use_qat=False, pretrained_path=None):
    config = toml.load(config_path)
    dataset_cfg = config["dataset"]
    model_cfg = config["model_template"]
    training = config["training"]

    timesteps = model_cfg["params"]["timesteps"]
    batch_size = training["batch_size"]
    epochs = training["epochs"]
    pretrain_epochs = None
    # QAT-FT: override epochs from [codegen].qat_ft_epochs if available
    codegen = config.get("codegen", {})
    if use_qat and pretrained_path and codegen.get("qat_ft_epochs"):
        epochs = int(codegen["qat_ft_epochs"])
        print(f"[QAT-FT] Overriding epochs to {epochs} (from codegen.qat_ft_epochs)")

    encoding = model_cfg["params"].get("encoding", None)
    dataset_name_lower = _ds_module_name(dataset_cfg["kind"])
    if encoding is None:
        encoding = _ds_default_encoding(dataset_cfg["kind"]) or "rate"
        print(f"[Info] Encoding auto-set to '{encoding}' for dataset '{dataset_name_lower}'")
    print(f"[Info] Encoding strategy: {encoding}")

    train_loader, test_loader, input_dim, num_classes = get_loaders(
        dataset_cfg, batch_size, timesteps, encoding=encoding
    )

    input_data_stats = profile_input_data(train_loader)
    print(f"[Info] Input data stats: min={input_data_stats['min']:.4f}, max={input_data_stats['max']:.4f}, "
          f"is_integer={input_data_stats['is_integer']}, is_signed={input_data_stats['is_signed']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind = model_cfg["kind"].lower()

    input_is_binary = detect_binary_input(train_loader, kind, timesteps, encoding)
    print(f"[INFO] Encoded input is_binary={input_is_binary} "
          f"(binary => first layer skips input multiply)")

    model = build_model(model_cfg, input_dim=input_dim, output_dim=num_classes)

    pretrained_acc = None
    pretrained_bn_already_folded = False
    if pretrained_path:
        print(f"[Pretrained] Loading weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # Prefer raw_state_dict (raw model keys) over standardized state_dict.
            state_dict = checkpoint.get("raw_state_dict") or checkpoint["state_dict"]
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                # FP32 checkpoint saved after BN fold: BN keys missing. Load with strict=False and skip the subsequent fold step.
                missing, _ = model.load_state_dict(state_dict, strict=False)
                bn_missing = [k for k in missing if "_bn_" in k or ".bn." in k]
                non_bn_missing = [k for k in missing if k not in bn_missing]
                if bn_missing and not non_bn_missing:
                    pretrained_bn_already_folded = True
                    print(f"[Pretrained] BN already folded in checkpoint "
                          f"({len(bn_missing)} BN keys skipped)")
                else:
                    raise
            pretrained_meta = checkpoint.get("training_meta", {})
            pretrained_acc = pretrained_meta.get("best_test_acc")
            if use_qat:
                pretrain_epochs = pretrained_meta.get("epochs")
            print(f"[Pretrained] Loaded model with best_test_acc={pretrained_acc:.2f}%"
                  if pretrained_acc else "[Pretrained] Loaded model")
        else:
            model.load_state_dict(checkpoint)
            print("[Pretrained] Loaded model state_dict")

    # Fold BN into weights before QAT so QAT learns scales on BN-folded weights.
    bn_was_folded = False
    if use_qat and getattr(model, 'use_bn', False) and not pretrained_bn_already_folded:
        model = fold_bn_into_model(model)
        bn_was_folded = True
        if not pretrained_path:
            print("[Warn] Direct QAT with use_bn=True: BN folded before training "
                  "(no-op on fresh model). Consider using QAT-FT for better accuracy.")
    elif pretrained_bn_already_folded:
        bn_was_folded = True
        for attr in ('conv_bn_layers', 'dw_bn_layers', 'pw_bn_layers', 'fc_bn_layers'):
            if hasattr(model, attr):
                delattr(model, attr)
        model.use_bn = False
        print("[Pretrained] Skipping BN fold (already folded in checkpoint), removed empty BN layers")

    qat_bit_width = (config.get("quantization", {}).get("bit_width")
                     or config.get("codegen", {}).get("quant_bits", 8))
    if use_qat:
        print(f"[QAT] Enabling Quantization-Aware Training with {qat_bit_width}-bit fake quantization")
        model = convert_to_qat_model(model, bit_width=qat_bit_width)

    model = model.to(device)

    base_lr = config["optimizer"]["lr"]
    if use_qat and pretrained_path:
        qat_lr_factor = config["optimizer"].get("qat_lr_factor", 0.1)
        lr = base_lr * qat_lr_factor
        print(f"[QAT] Using fine-tuning learning rate: {lr} (base_lr={base_lr} * factor={qat_lr_factor})")
    else:
        lr = base_lr

    optimizer, weight_decay = _build_optimizer(model, config, lr)
    scheduler_kind = training.get("scheduler", "none").lower()
    scheduler = _build_scheduler(
        optimizer, scheduler_kind, lr, epochs,
        len(train_loader), float(training.get("eta_min", 1e-6))
    )

    grad_clip = float(training.get("grad_clip", 0.0))
    if grad_clip > 0:
        print(f"[Info] Grad clipping: max_norm={grad_clip}")

    scheduler_per_epoch = scheduler_kind in ("cosine",)

    loss_type = training.get("loss", "ce_count")
    label_smoothing = float(training.get("label_smoothing", 0.0))

    training_history, best_state_dict = train_with_history(
        model, kind, train_loader, optimizer, device,
        epochs, timesteps, test_loader=test_loader, encoding=encoding,
        grad_clip=grad_clip, scheduler=scheduler,
        scheduler_per_epoch=scheduler_per_epoch,
        loss_type=loss_type, label_smoothing=label_smoothing
    )

    qat_scales = None
    if use_qat:
        if best_state_dict:
            # If BN was folded, rebuild model to match state_dict structure.
            if bn_was_folded:
                temp_cfg = dict(model_cfg)
                temp_cfg["params"] = dict(temp_cfg.get("params", {}))
                temp_cfg["params"]["use_bn"] = False
                temp_model = build_model(temp_cfg, input_dim=input_dim, output_dim=num_classes)
            else:
                temp_model = build_model(model_cfg, input_dim=input_dim, output_dim=num_classes)
            temp_model = convert_to_qat_model(temp_model, bit_width=qat_bit_width)
            temp_model.load_state_dict(best_state_dict)
            qat_scales = extract_qat_scales(temp_model)
            temp_model = convert_from_qat_model(temp_model)
            best_state_dict = temp_model.state_dict()
            print(f"[QAT] Extracted {len(qat_scales)} quantization scales from best model")
        else:
            qat_scales = extract_qat_scales(model)
            print(f"[QAT] Extracted {len(qat_scales)} quantization scales from final model")

        print("[QAT] Converting back to regular model for export...")
        model = convert_from_qat_model(model)

    os.makedirs("checkpoints", exist_ok=True)
    project_name = config.get("project", {}).get("name", f"{dataset_cfg['kind'].lower()}_{kind}")
    model_path = f"checkpoints/{project_name}.pth"

    if use_qat and pretrained_path:
        training_method = "QAT-FT"
    elif use_qat:
        training_method = "QAT"
    else:
        training_method = "FP32"

    # Non-QAT path: fold BN into weights for clean export.
    if not use_qat and getattr(model, 'use_bn', False):
        if best_state_dict:
            model.load_state_dict(best_state_dict)
        model = fold_bn_into_model(model)
        best_state_dict = model.state_dict()

    final_state_dict = best_state_dict if best_state_dict else model.state_dict()
    if best_state_dict:
        print(f"[Info] Saving best model (from best epoch)")
    else:
        print(f"[Info] Saving final model (no best model tracking)")

    # Produce standardized state_dict + layer_defs for export_ir.py.
    _std_cfg = dict(model_cfg)
    if not getattr(model, 'use_bn', True):
        # BN was folded: build fresh model without BN.
        _std_cfg = dict(model_cfg)
        _std_cfg["params"] = dict(_std_cfg.get("params", {}))
        _std_cfg["params"]["use_bn"] = False
    _fresh_model = build_model(_std_cfg, input_dim=input_dim, output_dim=num_classes)
    _fresh_model.load_state_dict(final_state_dict)
    _input_shape = input_dim if kind == "csnn" else None
    standardized_state_dict, layer_defs = _standardize_sd(_fresh_model, _input_shape)

    save_dict = {
        "state_dict": standardized_state_dict,  # standardized keys for export_ir.py
        "raw_state_dict": final_state_dict,      # raw model keys for pretrained loading
        "layer_defs": layer_defs,
        "model_kind": kind,
        "training_meta": {
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            "model_kind": kind,
            "training_method": training_method,
            "final_train_acc": training_history[-1]["train_acc"] if training_history else None,
            "final_test_acc": training_history[-1]["test_acc"] if training_history else None,
            "best_test_acc": max(h["test_acc"] for h in training_history) if training_history else None,
            "best_test_epoch": (max(range(len(training_history)),
                                    key=lambda i: training_history[i]["test_acc"]) + 1
                                if training_history else None),
            "weight_decay": weight_decay,
            "lr": lr,
            "base_lr": base_lr,
            "batch_size": batch_size,
            "pretrained_path": pretrained_path,
            "pretrained_acc": pretrained_acc,
        },
        "training_history": training_history,
        "qat_scales": qat_scales,  # QAT learned quantization scales (if any)
        "input_data_stats": input_data_stats,
        "encoding": encoding,
        "input_is_binary": input_is_binary,
    }
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")

    results_path = f"checkpoints/{project_name}_results.txt"
    with open(results_path, "w") as f:
        f.write(f"=== Training Results: {project_name} ===\n")
        f.write(f"Training method: {training_method}\n")
        if pretrained_path:
            f.write(f"Pretrained from: {pretrained_path}\n")
            if pretrained_acc:
                f.write(f"Pretrained acc: {pretrained_acc:.2f}%\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        if use_qat and pretrained_path:
            f.write(f"Base LR: {base_lr}, QAT factor: {config['optimizer'].get('qat_lr_factor', 0.1)}\n")
        f.write(f"Weight decay: {weight_decay}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"\n--- Epoch History ---\n")
        for i, h in enumerate(training_history):
            f.write(f"Epoch {i+1}: Train Acc={h['train_acc']:.2f}%, Test Acc={h['test_acc']:.2f}%\n")
        if training_history:
            best_epoch = max(range(len(training_history)),
                             key=lambda i: training_history[i]["test_acc"]) + 1
            best_acc = max(h["test_acc"] for h in training_history)
            f.write(f"\n--- Summary ---\n")
            f.write(f"Best Test Acc: {best_acc:.2f}% (Epoch {best_epoch})\n")
            f.write(f"Final Test Acc: {training_history[-1]['test_acc']:.2f}%\n")
    print(f"Training results saved to {results_path}")

    if use_qat:
        print(f"[QAT] Model trained with QAT, weights are quantization-friendly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN model from TOML config")
    parser.add_argument("config", help="Path to TOML config file")
    parser.add_argument("--qat", action="store_true", help="Enable Quantization-Aware Training (QAT)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint for QAT fine-tuning")
    args = parser.parse_args()

    main(args.config, use_qat=args.qat, pretrained_path=args.pretrained)
