import pandas as pd
import glob
import os, sys

def combine_parquet(file_pattern, output_file):
    files = glob.glob(file_pattern)

    df_list = [pd.read_parquet(f) for f in files]

    combined_df = pd.concat(df_list, ignore_index=True)

    if not output_file.endswith(".parquet"):
        output_file += ".parquet"

    combined_df.to_parquet(output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python combineParquet <input_file_pattern> <output_file>")
    else:
        combine_parquet(sys.argv[1], sys.argv[2])
