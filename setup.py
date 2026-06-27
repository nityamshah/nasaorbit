import wfdb
import json
import numpy as np
import re
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import entropy, skew, kurtosis

# Parse record for the patient id
def get_pid_from_record_name(record_name):
    match = re.match(r"TRM(\d+)-RHC\d+", record_name)
    if match is None:
        return None
    return int(match.group(1))

# Find available records to use
def find_available_records(processed_dir):
    processed_dir = Path(processed_dir)

    record_paths = []
    for hea_path in sorted(processed_dir.glob("TRM*-RHC*.hea")):
        record_base = hea_path.with_suffix("")   # remove .hea
        json_path = record_base.with_suffix(".json")
        dat_path = record_base.with_suffix(".dat")

        # Make sure matching .dat and .json exist
        if not dat_path.exists():
            print(f"Skipping {record_base.name}: missing .dat")
            continue

        if not json_path.exists():
            print(f"Skipping {record_base.name}: missing .json")
            continue

        record_paths.append(record_base)

    return record_paths

# Get CDecomp Label from json
def get_label_from_json(json_path):
    with open(json_path, "r") as f:
        meta = json.load(f)

    value = meta.get("CDecomp", None)

    try:
        value = int(value)
    except Exception:
        return None

    if value in [0, 1]:
        return value
    else:
        return None

def bandpass_filter(x, low=1, high=40, fs=500, order=3):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def detect_r_peaks(ecg, fs=500):
    # Simple R peak detector using thresholded peak finding
    ecg_filt = bandpass_filter(ecg, low=5, high=40, fs=fs, order=3)
    ecg_norm = ecg_filt / (np.std(ecg_filt) + 1e-8)
    min_distance = int(0.4 * fs)  # minimum 400ms between beats (around 150 bpm max)
    peaks, _ = find_peaks(ecg_norm, height=0.5, distance=min_distance)
    return peaks

def segment_beats(signal, r_peaks, fs=500, pre_ms=100, post_ms=400):
    # Cut signal into individual beats around R peaks
    pre = int(pre_ms * fs / 1000)
    post = int(post_ms * fs / 1000)
    beats = []
    for r in r_peaks:
        if r - pre >= 0 and r + post < len(signal):
            beats.append(signal[r - pre: r + post])
    return np.array(beats)  # shape is (n_beats, beat_len)

def time_features_beats(beats):
    # RMS feature commented out because performs better with just ptp, maintained in code for
    # future reference
    rms = np.sqrt(np.mean(beats**2, axis=1))
    ptp = np.ptp(beats, axis=1)
    
    features = []
    #for arr in [ptp, rms]:
    for arr in [ptp]:
        features += [np.mean(arr), np.std(arr), np.median(arr)]
    return features  # 6 features per axis

def freq_features_beats(beats, fs=500):
    # Some frequency features not used because performs better without,
    # maintained in code for future reference
    avg_beat = np.mean(beats, axis=0)
    f, pxx = welch(avg_beat, fs=fs, nperseg=min(256, len(avg_beat)))
    pxx_norm = pxx / (np.sum(pxx) + 1e-8)
    
    dom_freq = f[np.argmax(pxx)]
    spec_entropy = entropy(pxx_norm)
    bp1_10  = np.sum(pxx[(f >= 1)  & (f < 10)])
    bp10_20 = np.sum(pxx[(f >= 10) & (f < 20)])
    bp20_40 = np.sum(pxx[(f >= 20) & (f < 40)])
    
    #return [dom_freq, spec_entropy, bp1_10, bp10_20, bp20_40]  # 5 per axis
    return [bp10_20, dom_freq]

def timing_features(beats, fs=500, pre_ms=100):
    from scipy.ndimage import uniform_filter1d
    avg_beat = np.mean(beats, axis=0)
    smoothed = uniform_filter1d(avg_beat, size=10)

    r0 = int(pre_ms * fs / 1000)

    # Search only after R peak, e.g. 30–350 ms post-R
    start = r0 + int(0.03 * fs)
    end   = r0 + int(0.35 * fs)
    segment = smoothed[start:end]

    # Use abs because SCG polarity may differ by axis/placement
    peaks, _ = find_peaks(
        np.abs(segment),
        distance=int(0.05 * fs),
        prominence=0.2 * np.std(segment)
    )

    if len(peaks) >= 2:
        t1 = (start + peaks[0] - r0) / fs * 1000
        t2 = (start + peaks[1] - r0) / fs * 1000
        return [t1, t2 - t1]
    elif len(peaks) == 1:
        t1 = (start + peaks[0] - r0) / fs * 1000
        return [t1, 0.0]
    else:
        return [np.nan, np.nan]

def extract_patient_features(lat, hf, dv, ecg, fs=500):
    # Extract features at patient level (one feature vector per patient)
    r_peaks = detect_r_peaks(ecg, fs)
    
    if len(r_peaks) < 5:
        return None  # too few beats, skip
    
    features = []
    
    # Hf, dv axis commented out because performs better without,
    # maintained in code for future reference
    #for axis_signal in [lat, hf, dv]:
    for axis_signal in [lat]:
        beats = segment_beats(axis_signal, r_peaks, fs)
        if len(beats) < 3:
            return None
        
        # Raw amplitude features
        features += time_features_beats(beats)

        # Shape-normalized features
        beats_shape = (beats - np.mean(beats, axis=1, keepdims=True)) / (
            np.std(beats, axis=1, keepdims=True) + 1e-8
        )
        features += freq_features_beats(beats_shape, fs)
        features += timing_features(beats_shape, fs)

    return features

def run_pipeline():
    X, y, groups, record_names = [], [], [], []

    processed_dir = Path(r"C:\Users\nitya\Documents\VSCode\nasaorbitdatanewdownloads2\processed_data")
    record_paths = find_available_records(processed_dir)

    print(f"Found {len(record_paths)} available RHC records")

    for record_base in record_paths:
        record_name = record_base.name           
        pid = get_pid_from_record_name(record_name)

        if pid is None:
            print(f"Skipping {record_name}: could not parse PID")
            continue

        json_path = record_base.with_suffix(".json")
        label = get_label_from_json(json_path)

        if label is None:
            print(f"Skipping {record_name}: missing/invalid label")
            continue

        try:
            # wfdb.rdrecord wants the path without .hea/.dat extension
            record = wfdb.rdrecord(str(record_base))

            names = record.sig_name
            data = record.p_signal

            required_channels = [
                "patch_ACC_lat",
                "patch_ACC_hf",
                "patch_ACC_dv",
                "patch_ECG"
            ]

            missing = [ch for ch in required_channels if ch not in names]
            if missing:
                print(f"Skipping {record_name}: missing channels {missing}")
                continue

            lat = data[:, names.index("patch_ACC_lat")]
            hf  = data[:, names.index("patch_ACC_hf")]
            dv  = data[:, names.index("patch_ACC_dv")]
            ecg = data[:, names.index("patch_ECG")]

            lat = bandpass_filter(lat, 1, 40)
            hf  = bandpass_filter(hf,  1, 40)
            dv  = bandpass_filter(dv,  1, 40)
            ecg_filt = bandpass_filter(ecg, 0.5, 40)

            features = extract_patient_features(lat, hf, dv, ecg_filt)

            if features is None:
                print(f"Skipping {record_name}: feature extraction failed")
                continue

            X.append(features)
            y.append(label)

            # group by patient ID, NOT by RHC visit
            # This prevents RHC1 from a patient being in train while RHC2 is in test
            groups.append(pid)

            record_names.append(record_name)

            print(
                f"Processed {record_name} | PID: {pid} | "
                f"CDecomp label: {label} | Features: {len(features)}"
            )

        except Exception as e:
            print(f"Skipping {record_name}: error = {e}")
            continue

    return X, y, groups, record_names