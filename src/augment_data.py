
import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "augmented"

SEED = 42


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """Create one reproducible augmentation of an RGB image."""
    image = image.convert("RGB")

    # Butterfly orientation is often approximately symmetric:
    # horizontal flipping is a reasonable augmentation.
    image = ImageOps.mirror(image)

    # Apply a small rotation to improve robustness to orientation changes.
    angle = rng.uniform(-12.0, 12.0)
    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )

    return image


def generate_augmented_manifest(
    manifest_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    """Augment training images only and save a separate manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = {"absolute_path", "labels", "data set"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The source manifest is empty.")

    if df[list(required_columns)].isna().any().any():
        raise ValueError("The source manifest contains missing values.")

    train_df = df[df["data set"] == "train"].copy()

    if train_df.empty:
        raise ValueError("No training images found in the manifest.")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        train_df = train_df.head(limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(train_df)

    for row_number, (_, row) in enumerate(train_df.iterrows()):
        source_path = Path(row["absolute_path"])

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source image not found: {source_path}"
            )

        # A deterministic, independent random stream per source image.
        rng = random.Random(SEED + row_number * 1009)

        with Image.open(source_path) as source:
            image = source.convert("RGB")
            augmented = augment_image(image, rng)

        # Preserve the source extension-independent naming and avoid
        # overwriting the original dataset.
        output_path = (
            image_dir / f"train_{row_number:05d}_aug.jpg"
        )
        augmented.save(output_path, format="JPEG", quality=95)

        records.append(
            {
                "absolute_path": str(output_path.resolve()),
                "labels": str(row["labels"]),
                "data set": "train",
                "source_path": str(source_path.resolve()),
                "augmentation": "horizontal_flip_rotation",
            }
        )

        if (row_number + 1) % 500 == 0:
            print(f"Processed {row_number + 1}/{total} images")

    augmented_df = pd.DataFrame(records)

    # Validate the generated manifest before reporting success.
    if len(augmented_df) != len(train_df):
        raise RuntimeError(
            "Generated image count does not match the selected "
            "training image count."
        )

    if set(augmented_df["data set"].unique()) != {"train"}:
        raise RuntimeError(
            "The augmented manifest contains non-training samples."
        )

    original_classes = set(df["labels"].astype(str))
    augmented_classes = set(augmented_df["labels"])

    if not augmented_classes.issubset(original_classes):
        raise RuntimeError(
            "The augmented manifest contains unknown class labels."
        )

    missing_images = [
        path
        for path in augmented_df["absolute_path"]
        if not Path(path).is_file()
    ]

    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} generated image files are missing. "
            f"First missing file: {missing_images[0]}"
        )

    manifest_output = output_dir / "manifest.csv"
    augmented_df.to_csv(manifest_output, index=False)

    print("\nAugmentation completed successfully.")
    print(f"Original training images selected: {len(train_df)}")
    print(f"Generated augmented images: {len(augmented_df)}")
    print(f"Augmented manifest: {manifest_output}")
    print("\nAugmented images per class:")
    print(
        augmented_df["labels"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return augmented_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one augmented image per original training image."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the original processed manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated images and their manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for a small test run.",
    )

    args = parser.parse_args()

    manifest_path = args.manifest
    output_dir = args.output_dir

    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    generate_augmented_manifest(
        manifest_path=manifest_path.resolve(),
        output_dir=output_dir.resolve(),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
