import pandas as pd
import glob
import os, sys


import glob
import os
import sys
import pyarrow.parquet as pq
import pyarrow as pa

def combine_parquet(file_pattern, output_file):
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    if not output_file.endswith(".parquet"):
        output_file += ".parquet"

    print(f"Found {len(files)} files. Combining...")

    first_file = pq.ParquetFile(files[0])
    schema = first_file.schema.to_arrow_schema()

    with pq.ParquetWriter(output_file, schema, compression='zstd') as writer:
        for f in files:
            try:
                table = pq.read_table(f)
                writer.write_table(table)
            except Exception as e:
                print(f"Error processing file {f}: {e}", file=sys.stderr)

    print(f"Successfully created: {output_file}")


#def combine_parquet(file_pattern, output_file):
#    files = glob.glob(file_pattern)
#
#    df_list = [pd.read_parquet(f) for f in files]
#
#    combined_df = pd.concat(df_list, ignore_index=True)
#
#    if not output_file.endswith(".parquet"):
#        output_file += ".parquet"
#
#    combined_df.to_parquet(output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python combineParquet <input_file_pattern> <output_file>")
    else:
        combine_parquet(sys.argv[1], sys.argv[2])
