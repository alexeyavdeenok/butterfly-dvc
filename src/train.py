
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from dvclive import Live
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from tensorflow import keras

from image_dataset import build_datasets, load_manifest
from models import build_cnn, build_transfer_model, compile_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "params" / "experiments.yaml"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: Path) -> dict:
    """Load and validate the experiment YAML file."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    for section in ("data", "training"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")

    return config


def create_model(
    architecture: str,
    num_classes: int,
    dropout: float,
    image_size: tuple[int, int],
) -> keras.Model:
    """Construct the requested model architecture."""
    if architecture == "cnn":
        return build_cnn(
            num_classes=num_classes,
            dropout=dropout,
            image_size=image_size,
        )

    if architecture in ("mobilenetv2", "efficientnetb0"):
        return build_transfer_model(
            architecture=architecture,
            num_classes=num_classes,
            dropout=dropout,
            image_size=image_size,
        )

    raise ValueError(f"Unknown architecture: {architecture}")


def evaluate_model(
    model: keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
) -> dict:
    """Compute accuracy, macro F1 and a per-class report."""
    y_true = []
    y_pred = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 fingerprint for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_metadata() -> dict:
    """Capture the current Git revision and working-tree status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        return {
            "git_commit": commit,
            "git_working_tree_dirty": bool(status),
        }

    except (subprocess.SubprocessError, FileNotFoundError):
        return {
            "git_commit": None,
            "git_working_tree_dirty": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a butterfly classifier."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenetv2", "efficientnetb0"],
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment name.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the final selected checkpoint on the test split.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config["data"]
    training_config = config["training"]

    image_size_value = int(data_config.get("image_size", 224))
    image_size = (image_size_value, image_size_value)
    batch_size = int(data_config.get("batch_size", 32))
    epochs = int(training_config.get("epochs", 10))
    patience = int(training_config.get("early_stopping_patience", 3))
    learning_rate = float(
        training_config.get("learning_rate", 0.001)
    )
    dropout = float(training_config.get("dropout", 0.3))
    include_augmented = bool(data_config.get("augmentation", False))

    if epochs < 1:
        raise ValueError("training.epochs must be at least 1.")

    if patience < 0:
        raise ValueError(
            "training.early_stopping_patience cannot be negative."
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError("training.dropout must be in [0, 1).")

    if learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive.")

    model_name = args.model
    run_name = args.run_name or (
        f"{model_name}_{'augmented' if include_augmented else 'original'}"
    )

    output_dir = PROJECT_ROOT / "reports" / "experiments" / run_name
    model_dir = PROJECT_ROOT / "models" / run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(
        data_config.get(
            "manifest",
            "data/processed/manifest.csv",
        )
    )
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_path = manifest_path.resolve()

    # Validate the manifest before building TensorFlow datasets.
    manifest_df = load_manifest(manifest_path)
    original_train_count = int(
        (manifest_df["data set"] == "train").sum()
    )

    if include_augmented:
        augmented_manifest_path = (
            PROJECT_ROOT / "data" / "augmented" / "manifest.csv"
        )
        augmented_df = load_manifest(augmented_manifest_path)
        augmented_train_count = int(
            (augmented_df["data set"] == "train").sum()
        )
    else:
        augmented_train_count = 0

    datasets, class_names = build_datasets(
        batch_size=batch_size,
        image_size=image_size_value,
        seed=seed,
        manifest_path=manifest_path,
        include_augmented=include_augmented,
    )

    if len(class_names) != 100:
        raise ValueError(
            f"Expected 100 classes, got {len(class_names)}."
        )

    model = create_model(
        architecture=model_name,
        num_classes=len(class_names),
        dropout=dropout,
        image_size=image_size,
    )

    compile_model(
        model,
        learning_rate=learning_rate,
    )

    best_model_path = model_dir / "best.keras"
    started = time.perf_counter()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(output_dir / "training_history.csv")
        ),
    ]

    live_dir = output_dir / "dvclive"
    live_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Architecture: {model_name}")
    print(f"Classes: {len(class_names)}")
    print(f"Original training images: {original_train_count}")
    print(f"Augmented training images: {augmented_train_count}")
    print(f"Total training images: {original_train_count + augmented_train_count}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout: {dropout}")
    print(f"Epoch limit: {epochs}")
    print(f"Output directory: {output_dir}")
    print(f"Evaluate test split: {args.evaluate_test}")

    with Live(
        dir=str(live_dir),
        save_dvc_exp=False,
        report=None,
    ) as live:
        params_to_log = {
            "model": model_name,
            "run_name": run_name,
            "seed": seed,
            "image_size": image_size_value,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "early_stopping_patience": patience,
            "augmentation": include_augmented,
            "manifest_sha256": sha256_file(manifest_path),
            "evaluate_test": args.evaluate_test,
        }

        for key, value in params_to_log.items():
            live.log_param(key, value)

        class DVCLiveCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}

                for key, value in logs.items():
                    if value is not None:
                        live.log_metric(key, float(value))

                live.next_step()

        callbacks.append(DVCLiveCallback())

        history = model.fit(
            datasets["train"],
            validation_data=datasets["valid"],
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            # Training data is already shuffled by tf.data.
            shuffle=False,
        )

    elapsed_seconds = time.perf_counter() - started

    if not best_model_path.is_file():
        raise RuntimeError(
            f"Best checkpoint was not created: {best_model_path}"
        )

    # Evaluate the best checkpoint, not the final epoch by default.
    best_model = keras.models.load_model(best_model_path)

    validation_results = evaluate_model(
        best_model,
        datasets["valid"],
        class_names,
    )

    best_epoch = int(
        np.argmax(history.history["val_accuracy"]) + 1
    )

    final_results = {
        "run_name": run_name,
        "model": model_name,
        "augmentation": include_augmented,
        "seed": seed,
        "image_size": image_size_value,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": validation_results["accuracy"],
        "val_macro_f1": validation_results["macro_f1"],
        "original_train_images": original_train_count,
        "augmented_train_images": augmented_train_count,
        "total_train_images": (
            original_train_count + augmented_train_count
        ),
        "training_seconds": float(elapsed_seconds),
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "test_evaluated": args.evaluate_test,
        "class_names": class_names,
        **get_git_metadata(),
    }

    with (output_dir / "validation_classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            validation_results["classification_report"],
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Test metrics are only generated when explicitly requested.
    if args.evaluate_test:
        test_results = evaluate_model(
            best_model,
            datasets["test"],
            class_names,
        )

        final_results["test_accuracy"] = test_results["accuracy"]
        final_results["test_macro_f1"] = test_results["macro_f1"]

        with (output_dir / "test_classification_report.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                test_results["classification_report"],
                file,
                indent=2,
                ensure_ascii=False,
            )

    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            final_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nExperiment finished.")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
