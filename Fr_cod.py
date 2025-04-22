import pandas as pd

# File paths
file_path1 = r"C:\Users\rashe\Dropbox\1_Work\2_NITPY\1\1. aaTo Finish\4. J1 Low-Cost 9th  April\1. LTX_FD LCWQM_VS+DL\Fr_code\data_mac.csv"
file_path2 = r"C:\Users\rashe\Dropbox\1_Work\2_NITPY\1\1. aaTo Finish\4. J1 Low-Cost 9th  April\1. LTX_FD LCWQM_VS+DL\Fr_code\data_adak.csv"

# Read CSV files into dataframes
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
