"""
train_model.py

Phase 9: Trains a machine-learning fatigue classifier (Random Forest by
default, XGBoost optional) to replace the rule-based FatigueEngine
scoring logic.

Expects a labeled CSV dataset with the columns defined in
config.ML_FEATURE_COLUMNS plus a "label" column containing one of
config.ML_CLASS_LABELS.

Usage:
    python train_model.py --data path/to/dataset.csv

The trained model is saved to config.ML_MODEL_PATH via joblib, ready to
be picked up by FatigueEngine when config.USE_ML_MODEL is True.
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the labeled training dataset.

    Args:
        csv_path: Path to a CSV file containing feature columns and a
            "label" column.

    Returns:
        A validated pandas DataFrame.

    Raises:
        ValueError: If required columns are missing from the dataset.
    """
    df = pd.read_csv(csv_path)
    required_columns = set(config.ML_FEATURE_COLUMNS + ["label"])
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def train_random_forest(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the given dataset and print a
    held-out classification report.

    Args:
        df: DataFrame containing feature columns and a "label" column.

    Returns:
        The trained RandomForestClassifier.
    """
    x = df[config.ML_FEATURE_COLUMNS]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions)
    logger.info("Model evaluation:\n%s", report)
    print(report)

    return model


def main() -> None:
    """Parse CLI arguments, train the model, and save it to disk."""
    parser = argparse.ArgumentParser(
        description="Train the AI-DMS fatigue classifier."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the labeled training CSV file.",
    )
    args = parser.parse_args()

    logger.info("Loading dataset from %s", args.data)
    df = load_dataset(args.data)

    logger.info("Training RandomForestClassifier on %d samples...", len(df))
    model = train_random_forest(df)

    joblib.dump(model, config.ML_MODEL_PATH)
    logger.info("Trained model saved to %s", config.ML_MODEL_PATH)
    print(f"\nModel saved to: {config.ML_MODEL_PATH}")
    print("Set config.USE_ML_MODEL = True to activate it in the live app.")


if __name__ == "__main__":
    main()
