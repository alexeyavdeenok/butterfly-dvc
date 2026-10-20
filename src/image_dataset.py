from pathlib import Path

import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_manifest() -> pd.DataFrame:
    """Load the validated dataset manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. "
            "Run src/data_loader.py first."
        )

    df = pd.read_csv(MANIFEST_PATH)

    required = {"absolute_path", "labels", "data set"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    return df


def build_datasets(batch_size: int = BATCH_SIZE):
    """Create train, validation and test datasets."""
    df = load_manifest()

    class_names = sorted(df["labels"].unique().tolist())
    class_to_index = {name: i for i, name in enumerate(class_names)}

    datasets = {}

    for split in ("train", "valid", "test"):
        split_df = df[df["data set"] == split].copy()

        paths = split_df["absolute_path"].astype(str).to_numpy()
        labels = (
            split_df["labels"].map(class_to_index).to_numpy(dtype="int32")
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
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32)
            return image, label

        ds = ds.map(
            load_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if split == "train":
            ds = ds.shuffle(
                buffer_size=min(len(split_df), 2048),
                seed=SEED,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        datasets[split] = ds

    return datasets, class_names


def main():
    datasets, class_names = build_datasets()

    print(f"Number of classes: {len(class_names)}")

    for split, ds in datasets.items():
        images, labels = next(iter(ds))
        print(
            f"{split}: batch images={images.shape}, "
            f"batch labels={labels.shape}"
        )


if __name__ == "__main__":
    main()
