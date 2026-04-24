import wfdb
import json
import os
from collections import Counter

# Current patient IDs
# 39 patients
# NYHA class distribution: Class 1: 2 Class 2: 9 Class 3: 23 Class 4: 5
ids = [
    127, 128, 155, 158, 159, 164, 172,
    174, 176, 178, 181, 187, 195, 196, 198, 201, 203,
    215, 219, 220, 229, 231, 233, 235, 244, 245, 246,
    248, 252, 254, 255, 256, 260, 265, 267, 272, 275,
    278, 281
]

'''
#File Downloading
wfdb.dl_database("scg-rhc-wearable-database", "data")

# Downloaded .dat and .hea for ids below
# Still need .json for them

# Loop through and download JSON files
for i in ids:
    record = f"processed_data/TRM{i}-RHC1"
    
    wfdb.dl_files(
        "scg-rhc-wearable-database", "data",
        files=[f"{record}.json"]
    )
    
    print(f"Downloaded JSON for TRM{i}-RHC1")
'''
