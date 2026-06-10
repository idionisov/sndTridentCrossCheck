import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ROOT

def get_scifi_hit_density(scifi_hits, radius=40, min_check=False, min_hit_density=1000000):
    density = 0
    for hit in scifi_hits:
        channel_id = hit.GetChannelID()
        density += ROOT.snd.analysis_tools.densityScifi(channel_id, scifi_hits, radius, min_hit_density, min_check)
    return density

def get_scifi_total_qdc(scifi_hits):
    total_qdc = 0
    for hit in scifi_hits:
        total_qdc += hit.GetSignal()
    return total_qdc

def flush_to_parquet(data, batch_idx, cols, base_filename, verbose=False):
        df = pd.DataFrame(data, columns=cols)
        table = pa.Table.from_pandas(df)
        filename = f"{base_filename}_{batch_idx}.parquet"
        pq.write_table(table, filename, compression='snappy')

        if verbose:
            print(f"Batch {batch_idx} written to {filename}")
