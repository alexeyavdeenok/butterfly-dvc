
from pathlib import Path

import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
AUGMENTED_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "augmented" / "manifest.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

REQUIRED_COLUMNS = {"absolute_path", "labels", "data set"}
VALID_SPLITS = {"train", "valid", "test"}


def load_manifest(manifest_path=None) -> pd.DataFrame:
    """Load and validate a dataset manifest."""
    path = Path(manifest_path) if manifest_path else MANIFEST_PATH

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. "
            "Run the dataset validation/data loader step first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Manifest is empty: {path}")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    if df["labels"].isna().any():
        raise ValueError("Manifest contains missing class labels.")

    if df["absolute_path"].isna().any():
        raise ValueError("Manifest contains missing image paths.")

    df = df.copy()
    df["labels"] = df["labels"].astype(str)
    df["absolute_path"] = df["absolute_path"].astype(str)
    df["data set"] = df["data set"].astype(str)

    invalid_splits = set(df["data set"].unique()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(
            f"Unexpected split names in manifest: {sorted(invalid_splits)}"
        )

    # Check that all manifest entries point to existing files.
    missing_files = [
        file_path
        for file_path in df["absolute_path"]
        if not Path(file_path).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} image files are missing. "
            f"First missing file: {missing_files[0]}"
        )

    return df


def build_datasets(
    batch_size: int = BATCH_SIZE,
    image_size: int = 224,
    seed: int = SEED,
    manifest_path=None,
    include_augmented: bool = False,
):
    """Build train, validation and test tf.data datasets.

    Images are returned as float32 RGB tensors with pixel values in [0, 255].
    Model-specific normalization is handled inside the model.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if image_size < 1:
        raise ValueError("image_size must be at least 1.")

    df = load_manifest(manifest_path)

    original_class_names = sorted(df["labels"].unique().tolist())
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(original_class_names)
    }

    if include_augmented:
        augmented_df = load_manifest(AUGMENTED_MANIFEST_PATH)

        # Only training images from the augmented manifest are allowed.
        augmented_df = augmented_df[
            augmented_df["data set"] == "train"
        ].copy()

        if augmented_df.empty:
            raise ValueError(
                "The augmented manifest contains no training images."
            )

        augmented_classes = set(augmented_df["labels"].unique())
        original_classes = set(original_class_names)

        if not augmented_classes.issubset(original_classes):
            raise ValueError(
                "Augmented manifest contains unknown class labels."
            )

        # Prevent augmented samples from entering validation or test.
        df = pd.concat([df, augmented_df], ignore_index=True)

    datasets = {}
    image_shape = (image_size, image_size)

    for split in ("train", "valid", "test"):
        split_df = df[df["data set"] == split].copy()

        if split_df.empty:
            raise ValueError(f"No images found for split: {split}")

        paths = split_df["absolute_path"].to_numpy()
        labels = (
            split_df["labels"]
            .map(class_to_index)
            .to_numpy(dtype="int32")
        )

        if pd.isna(labels).any():
            raise ValueError(
                f"Some labels in split '{split}' could not be encoded."
            )

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def load_image(path, label):
            image_bytes = tf.io.read_file(path)

            image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False,
            )
            image.set_shape([None, None, 3])

            # Resize without changing the expected RGB channel order.
            image = tf.image.resize(image, image_shape)
            image = tf.cast(image, tf.float32)

            return image, label

        ds = ds.map(
            load_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if split == "train":
            ds = ds.shuffle(
                buffer_size=min(len(split_df), 2048),
                seed=seed,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        datasets[split] = ds

    return datasets, original_class_names


def main():
    datasets, class_names = build_datasets()

    print(f"Number of classes: {len(class_names)}")

    for split, dataset in datasets.items():
        images, labels = next(iter(dataset))
        print(
            f"{split}: images={images.shape}, "
            f"labels={labels.shape}, "
            f"pixel range=({float(tf.reduce_min(images)):.1f}, "
            f"{float(tf.reduce_max(images)):.1f})"
        )


if __name__ == "__main__":
    main()
