import pandas as pd

df = pd.read_csv("/eos/user/i/idioniso/tridents/all/sf3/recoTri.csv")
df['xz_z_vtx_12'] = (df['xz_c2'] - df['xz_c1']) / (df['xz_m1'] - df['xz_m2'])
df['xz_z_vtx_13'] = (df['xz_c3'] - df['xz_c1']) / (df['xz_m1'] - df['xz_m3'])
df['xz_z_vtx_23'] = (df['xz_c3'] - df['xz_c2']) / (df['xz_m2'] - df['xz_m3'])

df['yz_z_vtx_12'] = (df['yz_c2'] - df['yz_c1']) / (df['yz_m1'] - df['yz_m2'])
df['yz_z_vtx_13'] = (df['yz_c3'] - df['yz_c1']) / (df['yz_m1'] - df['yz_m3'])
df['yz_z_vtx_23'] = (df['yz_c3'] - df['yz_c2']) / (df['yz_m2'] - df['yz_m3'])

df.to_parquet("/eos/user/i/idioniso/tridents/all/sf3/recoTri.parquet", index=False)
