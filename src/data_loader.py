from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "archive"
CSV_PATH = DATA_ROOT / "butterflies and moths.csv"

EXPECTED_SPLITS = {"train", "valid", "test"}


def load_and_validate_data() -> pd.DataFrame:
    """Load the dataset manifest and validate image paths and labels."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = {"class id", "filepaths", "labels", "data set"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    if df[list(required_columns)].isnull().any().any():
        raise ValueError("The dataset CSV contains missing required values.")

    df["filepaths"] = df["filepaths"].astype(str)
    df["labels"] = df["labels"].astype(str)
    df["data set"] = df["data set"].astype(str)

    unknown_splits = set(df["data set"].unique()) - EXPECTED_SPLITS
    if unknown_splits:
        raise ValueError(f"Unexpected data splits: {sorted(unknown_splits)}")

    df["absolute_path"] = df["filepaths"].map(
        lambda path: (DATA_ROOT / path).resolve()
    )

    missing_files = df.loc[
        ~df["absolute_path"].map(Path.is_file), "filepaths"
    ]
    if not missing_files.empty:
        examples = missing_files.head(10).tolist()
        raise FileNotFoundError(
            f"{len(missing_files)} image files are missing. "
            f"Examples: {examples}"
        )

    if df["filepaths"].duplicated().any():
        duplicates = df.loc[
            df["filepaths"].duplicated(), "filepaths"
        ].head(10).tolist()
        raise ValueError(f"Duplicate image paths found: {duplicates}")

    classes_per_split = df.groupby("data set")["labels"].nunique()
    if (classes_per_split != 100).any():
        raise ValueError(
            "Expected all 100 classes in every split. "
            f"Found: {classes_per_split.to_dict()}"
        )

    return df


def main() -> None:
    df = load_and_validate_data()

    print("Dataset validation passed.")
    print(f"Total images: {len(df)}")
    print(f"Number of classes: {df['labels'].nunique()}")
    print("\nImages per split:")
    print(df["data set"].value_counts().sort_index().to_string())

    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
