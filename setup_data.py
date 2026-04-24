import wfdb
import json
import os
from collections import Counter
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import welch
from scipy.stats import entropy

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

def bandpass_filter(x, low=1, high=40, fs=500, order=3):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def get_signals(record):
    data = record.p_signal
    names = record.sig_name

    lat = data[:, names.index("Patch_ACC_lat")]
    hf = data[:, names.index("Patch_ACC_hf")]
    dv = data[:, names.index("Patch_ACC_dv")]

    return {"lat": lat, "hf": hf, "dv": dv}

def preprocess_scg(lat, hf, dv):

    # Convert the raw 3 axis SCG into magnitude
    scg = np.sqrt(lat**2 + hf**2 + dv**2)

    # Apply bandpass
    scg = bandpass_filter(scg, 1, 40)

    # Normalize
    scg = (scg - np.mean(scg)) / (np.std(scg) + 1e-8)

    return scg

def window_signal(signal, fs, win_sec, stride_sec):
    len_window = int(win_sec * fs)
    len_stride = int(stride_sec * fs)
    windows = []

    for start in range(0, len(signal) - len_window, len_stride):
        windows.append(signal[start:start + len_window])

    return windows

def get_label_for_record(record_name):
    path = f"C:/Users/nitya/Documents/VSCode/nasaorbitdata/processed_data/{record_name}.json"
    with open(path, "r") as file:
        meta = json.load(file)
    
    nyhac = meta["NYHAC"]

    return 0 if nyhac in [1, 2] else 1

def time_features(x):
    # mean, standard dev, rms, max, min, peak to peak
    return [
        np.mean(x), np.std(x), np.sqrt(np.mean(x**2)), np.max(x), np.min(x), np.ptp(x)
    ]

def freq_features(x, fs):
    f, pxx = welch(x, fs) #power spectral density
    dom_freq = f[np.argmax(pxx)]
    pxx_norm = pxx / (np.sum(pxx) + 1e-8)
    spec_entropy = entropy(pxx_norm)
    bp1_10 = np.sum(pxx[(f >= 1) & (f < 10)])
    bp10_20 = np.sum(pxx[(f >= 10) & (f < 20)])
    bp20_40 = np.sum(pxx[(f >= 20) & (f < 40)])

    return [dom_freq, spec_entropy, bp1_10, bp10_20, bp20_40]

def extract_features(window, fs):
    features = []

    features += time_features(window) 
    features += freq_features(window, fs)

    return features 

# Main starts here

X = []
y = []
groups = []

for pid in ids:
    record_name = f"TRM{pid}-RHC1"
    path = f"C:/Users/nitya/Documents/VSCode/nasaorbitdata/processed_data/{record_name}"

    record = wfdb.rdrecord(path)

    signals  = get_signals(record)

    scg = preprocess_scg(signals["lat"], signals["hf"], signals["dv"])

    windows = window_signal(scg, 500, 5, 2.5) # 500 is from the dataset

    label = get_label_for_record(record_name)

    for window in windows:
        features = extract_features(window, 500)
        X.append(features)
        y.append(label)
        groups.append(pid)

    print (f"Processed {record_name}")


