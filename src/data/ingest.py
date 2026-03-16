import pandas as pd
import yaml
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_lending_club(config: dict) -> pd.DataFrame:
    filepath = os.path.join(
        config["data"]["raw_dir"],
        config["data"]["lending_club_file"]
    )
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    return df

def basic_validation(df: pd.DataFrame) -> None:
    key_columns = [
        "loan_amnt", "int_rate", "annual_inc",
        "dti", "loan_status", "emp_length"
    ]
    missing = [col for col in key_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
    null_pct = df[key_columns].isnull().mean() * 100
    print("\nNull % in key columns:")
    print(null_pct.round(2).to_string())

if __name__ == "__main__":
    config = load_config()
    df = load_lending_club(config)
    basic_validation(df)
    print("\nIngestion check passed.")