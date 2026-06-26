import ROOT
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

if not hasattr(ROOT, 'sndScifiHit'):
    for library in ["libBase", "libShipData", "libshipLHC", "libsnd_analysis_tools"]:
        ROOT.gSystem.Load(library)




def run_per_plane_extraction(input_files, output_parquet, tree_name="rawConv"):
    chain = ROOT.TChain(tree_name)
    for f in input_files:
        chain.Add(f)

    total_entries = chain.GetEntries()
    print(f"Total entries to process: {total_entries}")

    data = []
    batch_size = 10000
    batch_counter = 0

    for i in range(total_entries):
        if i % 1000 == 0:
            print(f"Processing entry {i}/{total_entries}")

        chain.GetEntry(i)
        event_number = chain.EventHeader.GetEventNumber()
        scifi_hits = chain.Digi_ScifiHits

        planes = {}
        for s in range(1, 6):
            for v in [0, 1]:
                planes[(s, v)] = {'nhits': 0, 'qdc': 0.0}

        for hit in scifi_hits:
            if not hit.isValid(): continue
            station = hit.GetStation()
            is_vertical = 1 if hit.isVertical() else 0

            if (station, is_vertical) in planes:
                planes[(station, is_vertical)]['nhits'] += 1
                planes[(station, is_vertical)]['qdc'] += hit.GetSignal()

        for (station, is_vertical), vals in planes.items():
            if vals['nhits'] > 0:
                data.append({
                    'event_number': event_number,
                    'plane_id': f"S{station}{'V' if is_vertical else 'H'}",
                    'nhits': vals['nhits'],
                    'qdc': vals['qdc']
                })

        if len(data) >= batch_size:
            flush_batch(data, output_parquet, batch_counter)
            data = []
            batch_counter += 1

    if data:
        flush_batch(data, output_parquet, batch_counter)


def flush_batch(data, base_filename, batch_idx):
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    path = Path(base_filename)
    filename = f"{path.with_suffix('')}_{batch_idx}.parquet"
    pq.write_table(table, filename, compression='snappy')
    print(f"Saved {filename}")



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python perPlane.py <output_parquet> <input_file1> <input_file2> ...")
    else:
        output_name = sys.argv[1]
        inputs = sys.argv[2:]
        run_per_plane_extraction(inputs, output_name)
