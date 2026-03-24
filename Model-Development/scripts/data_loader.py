"""
data_loader.py — Loads the processed dataset produced by the Data Pipeline.

Expected input:
    ../Data-Pipeline/data/processed/application_train_merged.pkl

Returns:
    pd.DataFrame ready for preprocessing.
"""

import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_loader.log"),
        logging.StreamHandler()
    ]
)

# Path to the merged dataset produced by the Data Pipeline.
# From scripts/ we go up to Model-Development/, then up to
# shift-happens/, then into Data-Pipeline/data/processed/.
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "Data-Pipeline", "data", "processed", "application_train_merged.pkl"
)


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the merged pickle dataset into a DataFrame.

    Args:
        filepath: Path to the .pkl file. Defaults to
                  ../Data-Pipeline/data/processed/application_train_merged.pkl

    Returns:
        pd.DataFrame containing the full merged dataset.

    Raises:
        FileNotFoundError if the dataset does not exist at the given path.
    """
    logging.info(f"Loading dataset from: {filepath}")

    if not os.path.exists(filepath):
        logging.error(f"Dataset not found: {filepath}")
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Ensure the Data Pipeline has been run first."
        )

    df = pd.read_pickle(filepath)
    logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns[:10])}... ({len(df.columns)} total)")
    logging.info(f"TARGET distribution:\n{df['TARGET'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("ShiftHappens — Data Loader")
    logging.info("=" * 60)

    df = load_data()

    logging.info(f"Shape: {df.shape}")
    logging.info(f"Dtypes:\n{df.dtypes.value_counts().to_string()}")
    logging.info("Data loading complete.")
