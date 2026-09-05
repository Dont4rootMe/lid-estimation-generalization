"""Small Hydra-only VP learning diagnostic, not the full benchmark campaign.

Fits a fixed examples-seen budget; chooses a checkpoint by target-free loss
and lambda by LID MAE on the same optimizer-disjoint source-train holdout.
Only after freezing lambda are validation/test generated and evaluated once.
Regenerated coefficient geometries are never labelled canonical paper data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from datasets.vp_pilot import PILOT_LIDS, PINNED_UPSTREAM_REVISION, sample_vp_fixture
from models.readouts import diffusion_flipd
from models.training import _atomic_torch_save
from models.vp_baseline import (
    VPSchedule,
    VPBottleneckMLP,
    denoise,
    vp_loss,
    vp_primitives,
)
from utils.provenance import verify_upstream_source


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_config(config: dict) -> None:
    expected = {
        "schema_version",
        "seed",
        "dataset",
        "device",
        "output_root",
        "data",
        "model",
        "training",
        "evaluation",
    }
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("unsupported VP pilot configuration schema")
    fields = {
        "data": {
            "provenance_class",
            "source_train_count",
            "selection_count",
            "evaluation_count",
            "ambient_dim",
            "gaussian_control_lid",
            "normalization",
        },
        "model": {"hidden_sizes", "time_dim", "beta_min", "beta_max"},
        "training": {
            "steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "validation_interval",
            "early_stopping",
            "deterministic",
            "sampling",
            "loss",
            "checkpoint_selection",
        },
        "evaluation": {
            "batch_size",
            "lambda_min_power2",
            "lambda_max_power2",
            "grid_count",
            "trace_backend",
            "boundary_policy",
            "learning_noise_ratios",
        },
    }
    for section, names in fields.items():
        if set(config[section]) != names:
            raise ValueError(f"unexpected {section} fields")
    for section, key, value in (
        (
            "data",
            "provenance_class",
            "regenerated_known_lid_pilot_not_canonical_archive",
        ),
        ("data", "normalization", "train_fit_mean_global_rms"),
        ("training", "early_stopping", False),
        ("training", "deterministic", True),
        ("training", "sampling", "uniform_with_replacement"),
        ("training", "loss", "feature_sum_negative_noise_mse"),
        ("training", "checkpoint_selection", "minimum_train_selection_native_loss"),
        ("evaluation", "trace_backend", "exact"),
        ("evaluation", "boundary_policy", "report_unresolved_do_not_evaluate_test"),
    ):
        if config[section][key] != value:
            raise ValueError(f"{section}.{key} must equal {value!r}")
    if config["dataset"] not in PILOT_LIDS:
        raise ValueError("unsupported known-LID pilot dataset")
    data, tr, ev = config["data"], config["training"], config["evaluation"]
    for value in (
        data["source_train_count"],
        data["selection_count"],
        data["evaluation_count"],
        tr["steps"],
        tr["batch_size"],
        tr["validation_interval"],
        ev["batch_size"],
        ev["grid_count"],
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("counts must be positive integers")
    if data["ambient_dim"] != 30 or data["gaussian_control_lid"] != 3:
        raise ValueError("invalid known-LID data dimension")
    if not 2 <= data["selection_count"] < data["source_train_count"]:
        raise ValueError("selection subset must leave a nonempty optimizer fit set")
    if not 0 <= tr["warmup_steps"] < tr["steps"]:
        raise ValueError("warmup must be shorter than training")
    if ev["grid_count"] < 3 or ev["lambda_min_power2"] >= ev["lambda_max_power2"]:
        raise ValueError("invalid selection grid")
    if not math.isfinite(tr["learning_rate"]) or tr["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(tr["weight_decay"]) or tr["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not ev["learning_noise_ratios"]:
        raise ValueError("learning diagnostics require at least one scale")
    with torch.device("meta"):
        VPBottleneckMLP(
            data["ambient_dim"],
            tuple(config["model"]["hidden_sizes"]),
            config["model"]["time_dim"],
        )
    schedule = VPSchedule(config["model"]["beta_min"], config["model"]["beta_max"])
    schedule.time_for_lambda(2.0 ** ev["lambda_max_power2"])
    for value in ev["learning_noise_ratios"]:
        schedule.time_for_lambda(value)


def generate(config: dict, count: int, seed: int) -> np.ndarray:
    return sample_vp_fixture(config["dataset"], count, seed)


def native_loss(model, data, schedule, seed, batch_size):
    gen = torch.Generator(device=data.device).manual_seed(seed)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch in data.split(batch_size):
            t = torch.rand(len(batch), device=data.device, generator=gen)
            noise = torch.randn(batch.shape, device=data.device, generator=gen)
            total += float(vp_loss(model, batch, t, noise, schedule)) * len(batch)
    return total / len(data)


def predict(model, data, lam, schedule, batch_size):
    arrays, responses, corrections = [], [], []
    for batch in data.split(batch_size):
        p = vp_primitives(model, batch, float(lam), schedule=schedule)
        score = p.score.detach().cpu().double().numpy()
        divergence = p.score_divergence.detach().cpu().double().numpy()
        arrays.append(
            diffusion_flipd(score, divergence, sigma=p.sigma, ambient_dim=data.shape[1])
        )
        responses.append(data.shape[1] + p.sigma**2 * divergence)
        corrections.append(p.sigma**2 * np.sum(score**2, axis=1))
    return tuple(np.concatenate(x) for x in (arrays, responses, corrections))


def _run(config: dict) -> dict:
    validate_config(config)
    root = Path(__file__).resolve().parents[1]
    if verify_upstream_source(root / "lid_benchmarks") != PINNED_UPSTREAM_REVISION:
        raise ValueError("upstream source revision mismatch")
    destination = (
        Path(config["output_root"]) / f"{config['dataset']}-seed-{config['seed']}"
    )
    # A pilot is not resumable. Never overwrite another run/checkpoint.
    destination.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config), destination / "resolved_config.yaml")
    seed = int(config["seed"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data_cfg, tr, ev = config["data"], config["training"], config["evaluation"]
    source = generate(config, data_cfg["source_train_count"], seed)
    order = np.random.default_rng(seed + 1).permutation(len(source))
    selection_indices = np.sort(order[: data_cfg["selection_count"]])
    fit_indices = np.sort(order[data_cfg["selection_count"] :])
    mean = source[fit_indices].mean(axis=0, dtype=np.float64)
    rms = float(np.sqrt(np.mean((source[fit_indices].astype(np.float64) - mean) ** 2)))
    if not math.isfinite(rms) or rms <= 1e-8:
        raise ValueError("degenerate normalization")
    fit = torch.from_numpy(((source[fit_indices] - mean) / rms).astype(np.float32)).to(
        device
    )
    selection = torch.from_numpy(
        ((source[selection_indices] - mean) / rms).astype(np.float32)
    ).to(device)
    np.savez(
        destination / "source_train.npz",
        source=source,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        mean=mean,
        rms=rms,
    )
    known_lid = PILOT_LIDS[config["dataset"]]
    model_cfg = config["model"]
    schedule = VPSchedule(model_cfg["beta_min"], model_cfg["beta_max"])
    model = VPBottleneckMLP(
        30, tuple(model_cfg["hidden_sizes"]), model_cfg["time_dim"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"]
    )
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    initial_loss = native_loss(model, selection, schedule, seed + 3, ev["batch_size"])
    history, best_loss, best_step = [], math.inf, 0
    started = time.monotonic()
    for step in range(1, tr["steps"] + 1):
        model.train()
        indices = torch.randint(
            len(fit), (tr["batch_size"],), device=device, generator=generator
        )
        batch = fit[indices]
        times = torch.rand(len(batch), device=device, generator=generator)
        noise = torch.randn(batch.shape, device=device, generator=generator)
        multiplier = min(1.0, step / max(tr["warmup_steps"], 1))
        if step > tr["warmup_steps"]:
            multiplier = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (step - tr["warmup_steps"])
                    / (tr["steps"] - tr["warmup_steps"])
                )
            )
        optimizer.param_groups[0]["lr"] = tr["learning_rate"] * multiplier
        optimizer.zero_grad(set_to_none=True)
        loss = vp_loss(model, batch, times, noise, schedule)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        if step % tr["validation_interval"] == 0 or step == tr["steps"]:
            heldout = native_loss(
                model, selection, schedule, seed + 3, ev["batch_size"]
            )
            if not math.isfinite(heldout):
                raise FloatingPointError("non-finite held-out native loss")
            record = {
                "step": step,
                "examples_seen": step * tr["batch_size"],
                "train_native_loss": float(loss.detach()),
                "selection_native_loss": heldout,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            if heldout < best_loss:
                best_loss, best_step = heldout, step
                _atomic_torch_save(
                    destination / "best.pt",
                    {
                        "schema_version": 1,
                        "contract": schedule.contract(),
                        "config": config,
                        "step": step,
                        "examples_seen": step * tr["batch_size"],
                        "model_state": {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                        "normalization_mean": torch.from_numpy(mean),
                        "normalization_rms": rms,
                    },
                )
            write_json(destination / "history.json", history)
            write_json(
                destination / "status.json",
                {
                    "status": "training",
                    **record,
                    "best_step": best_step,
                    "best_native_loss": best_loss,
                },
            )
            print(json.dumps({"dataset": config["dataset"], **record}), flush=True)
    training_seconds = time.monotonic() - started
    _atomic_torch_save(
        destination / "last.pt",
        {
            "schema_version": 1,
            "contract": schedule.contract(),
            "config": config,
            "step": tr["steps"],
            "examples_seen": tr["steps"] * tr["batch_size"],
            "model_state": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "normalization_mean": torch.from_numpy(mean),
            "normalization_rms": rms,
        },
    )
    loaded = torch.load(destination / "best.pt", map_location=device, weights_only=True)
    if loaded["contract"] != schedule.contract() or loaded["config"] != config:
        raise ValueError("checkpoint contract mismatch")
    model.load_state_dict(loaded["model_state"], strict=True)
    model.eval()
    quality = []
    with torch.no_grad():
        noise_gen = torch.Generator(device=device).manual_seed(seed + 4)
        noise = torch.randn(selection.shape, device=device, generator=noise_gen)
        for lam in ev["learning_noise_ratios"]:
            t = torch.full(
                (len(selection),), schedule.time_for_lambda(lam), device=device
            )
            alpha, sigma = schedule.coefficients(t)
            noisy = alpha[:, None] * selection + sigma[:, None] * noise
            predicted = denoise(model, noisy, t, schedule)
            mse = float((predicted - selection).square().mean())
            identity_mse = float((noisy / alpha[:, None] - selection).square().mean())
            zero_mse = float(selection.square().mean())
            quality.append(
                {
                    "lambda": lam,
                    "time": float(t[0]),
                    "reconstruction_mse": mse,
                    "identity_mse": identity_mse,
                    "train_mean_mse": zero_mse,
                    "ratio_to_better_trivial": mse / min(identity_mse, zero_mse),
                }
            )
        np.savez(
            destination / "denoising_projection.npz",
            noise_ratio=lam,
            clean=selection.cpu().numpy(),
            noisy=(noisy / alpha[:, None]).cpu().numpy(),
            denoised=predicted.cpu().numpy(),
        )
    write_json(destination / "learning_quality.json", quality)
    lambdas = np.exp2(
        np.linspace(ev["lambda_min_power2"], ev["lambda_max_power2"], ev["grid_count"])
    )
    columns = [
        predict(model, selection, lam, schedule, ev["batch_size"]) for lam in lambdas
    ]
    full, response, correction = (
        np.stack([col[i] for col in columns], axis=1) for i in range(3)
    )
    if not np.isfinite(full).all():
        raise FloatingPointError("non-finite selection curve")
    np.savez(
        destination / "selection_curve.npz",
        lambdas=lambdas,
        full=full,
        response=response,
        correction=correction,
        target=np.full(len(selection), known_lid),
    )
    maes = np.mean(np.abs(full - known_lid), axis=0)
    selected = int(np.flatnonzero(maes <= maes.min() + 1e-12)[0])
    boundary = selected in (0, len(lambdas) - 1)
    selected_record = {
        "index": selected,
        "lambda": float(lambdas[selected]),
        "native_time": schedule.time_for_lambda(float(lambdas[selected])),
        "selection_mae": float(maes[selected]),
        "boundary": boundary,
        "test_accessed_at_freeze": False,
    }
    write_json(destination / "frozen_selection.json", selected_record)
    evaluation = {}
    if not boundary:
        for offset, split in ((10, "validation"), (11, "test")):
            raw = generate(config, data_cfg["evaluation_count"], seed + offset)
            tensor = torch.from_numpy(((raw - mean) / rms).astype(np.float32)).to(
                device
            )
            preds, resp, corr = predict(
                model, tensor, lambdas[selected], schedule, ev["batch_size"]
            )
            if not np.isfinite(preds).all():
                raise FloatingPointError(f"non-finite {split} predictions")
            np.savez(
                destination / f"{split}.npz",
                raw=raw,
                full=preds,
                response=resp,
                correction=corr,
                target=np.full(len(raw), known_lid),
            )
            evaluation[split] = {
                "mae": float(np.mean(np.abs(preds - known_lid))),
                "mean_lid": float(preds.mean()),
                "std_lid": float(preds.std()),
                "finite_fraction": float(np.isfinite(preds).mean()),
            }
    summary = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "provenance_class": data_cfg["provenance_class"],
        "upstream_revision": PINNED_UPSTREAM_REVISION,
        "true_lid": known_lid,
        "ambient_dim": 30,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "examples_seen": tr["steps"] * tr["batch_size"],
        "best_checkpoint_examples_seen": best_step * tr["batch_size"],
        "initial_native_loss": initial_loss,
        "best_native_loss": best_loss,
        "native_loss_ratio": best_loss / initial_loss,
        "training_seconds": training_seconds,
        "normalization_rms": rms,
        "selected": selected_record,
        "evaluation": evaluation,
        "quality": quality,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(destination / "summary.json", summary)
    write_json(
        destination / "status.json",
        {
            "status": "complete_boundary_unresolved" if boundary else "complete",
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    source_paths = [
        "models/vp_baseline.py",
        "models/neural_fields.py",
        "models/readouts.py",
        "models/training.py",
        "experiments/vp_baseline_pilot.py",
        "datasets/vp_pilot.py",
        "configs/vp_baseline_pilot.yaml",
        "uv.lock",
    ]
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "source_sha256": {name: file_sha(root / name) for name in source_paths},
            "outputs_sha256": {
                p.name: file_sha(p)
                for p in sorted(destination.iterdir())
                if p.is_file()
            },
            "environment": {
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "tf32": False,
            },
            "evidence_level": "small_regenerated_VP_pilot_not_full_benchmark",
        },
    )
    print(json.dumps(summary), flush=True)
    return summary


def run(config: dict) -> dict:
    """Run an isolated pilot without leaving changed torch flags in callers."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        return _run(config)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@hydra.main(version_base="1.3", config_path=None, config_name="vp_baseline_pilot")
def _hydra_main(config: DictConfig) -> None:
    run(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))


def main() -> None:
    """Discover YAML with the same source/wheel contract as the main CLI."""
    from experiments.cli import _default_config_dir

    has_config_dir = any(
        argument == "--config-dir" or argument.startswith("--config-dir=")
        for argument in sys.argv[1:]
    )
    if not has_config_dir:
        sys.argv[1:1] = ["--config-dir", str(_default_config_dir())]
    _hydra_main()


if __name__ == "__main__":
    main()
