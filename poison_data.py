import pandas as pd
import numpy as np
import argparse

def poison_data(input_file, output_file, percentage):
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    target_column = 'species'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")
    unique_classes = df[target_column].unique()
    if len(unique_classes) < 2:
        df.to_csv(output_file, index=False)
        return
    n_rows_to_poison = int(len(df) * (percentage / 100.0))
    if n_rows_to_poison == 0:
        df.to_csv(output_file, index=False)
        return
    print(f"Poisoning {n_rows_to_poison} rows ({percentage}%)...")
    poison_indices = np.random.choice(df.index, size=n_rows_to_poison, replace=False)
    for idx in poison_indices:
        current_label = df.loc[idx, target_column]
        other_labels = [label for label in unique_classes if label != current_label]
        new_label = np.random.choice(other_labels)
        df.loc[idx, target_column] = new_label
    print(f"Saving poisoned data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a dataset by mislabeling a percentage of rows.")
    parser.add_argument("--input-file", required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--output-file", required=True, help="Path to save the poisoned CSV dataset.")
    parser.add_argument("--percentage", type=int, required=True, help="Percentage of data to poison (0-100).")
    args = parser.parse_args()
    poison_data(args.input_file, args.output_file, args.percentage)
