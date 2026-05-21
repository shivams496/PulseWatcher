import wfdb
import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data"
RECORD_IDS = ['100', '101', '103', '105', '106', '107',
               '108', '109', '111', '112', '113', '114',
               '115', '116', '117', '118', '119', '121',
               '122', '123', '124', '200', '201', '202',
               '203', '205', '207', '208', '209', '210',
               '212', '213', '214', '215', '217', '219',
               '220', '221', '222', '223', '228', '230',
               '231', '232', '233', '234']
WINDOW_SIZE = 187
NORMAL_LABEL = 'N'


def download_data():
    """Downloads MIT-BIH records from PhysioNet into data/ folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {len(RECORD_IDS)} records from PhysioNet...")

    for record_id in RECORD_IDS:
        wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[record_id])
        print(f"  Downloaded record {record_id}")

    print("Download complete.")


def load_record(record_id):
    """Loads one patient's ECG signal and its beat annotations."""
    record_path = os.path.join(DATA_DIR, record_id)
    record = wfdb.rdrecord(record_path, channels=[0])
    annotation = wfdb.rdann(record_path, 'atr')
    signal = record.p_signal[:, 0]
    return signal, annotation


def segment_beats(signal, annotation):
    """
    Cuts the continuous ECG signal into individual beat windows.
    Each beat is centered on its R-peak and is WINDOW_SIZE long.
    """
    beats = []
    labels = []
    half = WINDOW_SIZE // 2

    for i, peak in enumerate(annotation.sample):
        label = annotation.symbol[i]
        start = peak - half
        end = peak + half + 1

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        if len(beat) != WINDOW_SIZE:
            continue

        beats.append(beat)
        labels.append(label)

    return np.array(beats), np.array(labels)


def normalize(beats):
    """Scales each beat to the range [0, 1]."""
    beats_min = beats.min(axis=1, keepdims=True)
    beats_max = beats.max(axis=1, keepdims=True)
    normalized = (beats - beats_min) / (beats_max - beats_min + 1e-8)
    return normalized


def preprocess_and_save():
    """Main function: loads all records, segments, normalizes, and saves."""
    all_normal = []
    all_anomalous = []

    for record_id in RECORD_IDS:
        print(f"Processing record {record_id}...")
        try:
            signal, annotation = load_record(record_id)
            beats, labels = segment_beats(signal, annotation)
            beats = normalize(beats)

            normal_mask = labels == NORMAL_LABEL
            all_normal.append(beats[normal_mask])
            all_anomalous.append(beats[~normal_mask])

        except Exception as e:
            print(f"  Skipping record {record_id}: {e}")
            continue

    normal_beats = np.vstack(all_normal)
    anomalous_beats = np.vstack(all_anomalous)

    print(f"\nTotal normal beats: {len(normal_beats)}")
    print(f"Total anomalous beats: {len(anomalous_beats)}")
    np.random.shuffle(normal_beats)

    split = int(0.8 * len(normal_beats))
    train = normal_beats[:split]
    test = normal_beats[split:]

    np.save(os.path.join(DATA_DIR, 'train.npy'), train)
    np.save(os.path.join(DATA_DIR, 'test.npy'), test)
    np.save(os.path.join(DATA_DIR, 'anomaly.npy'), anomalous_beats)

    print(f"Saved: train={train.shape}, test={test.shape}, anomaly={anomalous_beats.shape}")


if __name__ == "__main__":
    download_data()
    preprocess_and_save()
    