import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy import genfromtxt
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import convert_accelerometer_for_android, convert_gyroscope_for_android, kalman, plot_sensor_data, \
    convert_magnetometer_for_android, resample_sequence, convert_barometer_for_android


class FallAllDDataset(Dataset):

    visualization = True

    def __init__(self, root_dir, accellerometer=True, gyroscope=True, magnetometer=False, barometer=False):
        self.root_dir = root_dir
        self.samples = []
        self.windows = []
        self.sampleClasses = {0: 0, 1: 0}
        self.windowsClasses = {0: 0, 1: 0}


        data = defaultdict(dict)

        for file in tqdm(os.listdir(root_dir), desc="Processing files"):
            match = re.match(r"^S(\d{2})_D(\d+)_A(\d{3})_T(\d{2})_([A-Z])\.dat$", file)
            if not match:
                print(f'File name {file} does not match the expected pattern.')
                continue
            subject_id = match.group(1)
            device_id = match.group(2)
            activity_id = match.group(3)
            trial_no = match.group(4)
            sensor = match.group(5)
            file = os.path.join(root_dir, file)


            if int(device_id) == 3:
                chiave = (subject_id, device_id, activity_id, trial_no)
                data[chiave]['subject'] = subject_id
                data[chiave]['device'] = int(device_id)
                data[chiave]['activity'] = activity_id
                data[chiave]['trial'] = trial_no
                data[chiave]['label'] = 1 if int(activity_id) >= 101 else 0
                if sensor == 'A' and accellerometer:
                    data[chiave][sensor] = kalman(convert_accelerometer_for_android(resample_sequence(genfromtxt(file, delimiter=','), original_hz=238, target_hz=20)))
                elif sensor == 'G' and gyroscope:
                    data[chiave][sensor] = kalman(convert_gyroscope_for_android(resample_sequence(genfromtxt(file, delimiter=','), original_hz=238, target_hz=20)))
                elif sensor == 'M' and magnetometer:
                    data[chiave][sensor] = kalman(convert_magnetometer_for_android(resample_sequence(genfromtxt(file, delimiter=','), original_hz=80, target_hz=20)))
                elif sensor == 'B' and barometer:
                    data[chiave][sensor] = convert_barometer_for_android(genfromtxt(file, delimiter=','))

        self.samples = list(data.values())

        for sample in self.samples:
            if sample['device'] == 3:
                data = self.__merge_sample_sensors(
                    accel=sample['A'] if accellerometer else None,
                    gyro=sample['G'] if gyroscope else None,
                    magn=sample['M'] if magnetometer else None,
                    baro=sample['B'] if barometer else None)
                windows = self.__windowing_with_overlap__(data.values, (sample['subject'], sample['activity'], sample['trial']), sample['label'], window_size_s=7, fs=20, overlap=0.5, impact_time_s=10)
                self.windows.extend(windows)
                if sample['label'] == 0:
                    self.sampleClasses[0] += 1
                else:
                    self.sampleClasses[1] += 1

    def __len__(self):
        if self.visualization:
            return len(self.samples)

        return len(self.windows)

    def __getitem__(self, idx):
        if self.visualization:
            return self.samples[idx]

        features = self.windows[idx][1].astype(np.float32)
        return features, self.windows[idx][2]

    def __windowing_with_overlap__(self, data, info, label, window_size_s=7, fs=20, overlap=0.5, impact_time_s=10):
        window_length = int(window_size_s * fs)
        stride = int(window_length * (1 - overlap))
        num_samples = data.shape[0]

        segments_list = []  # indice del punto centrale della caduta in campioni

        impact_idx = int(impact_time_s * fs)  # start indices per sliding window

        start_indices = range(0, num_samples - window_length + 1, stride)

        for start in start_indices:

            end = start + window_length
            window = data[start:end, :].T  # shape (3, window_length)
            if label == 0:
                # ADL → tutti i segmenti
                segments_list.append((info, window, 0))
                self.windowsClasses[0] += 1
            else:
                # Fall → solo se la finestra contiene il punto di impatto
                if start <= impact_idx < end:
                    segments_list.append((info, window, 1))
                    self.windowsClasses[1] += 1
        return segments_list

    def classes(self):
        if self.visualization:
            return self.sampleClasses
        return self.windowsClasses

    def setVisualization(self, visualization):
        self.visualization = visualization

    def __merge_sample_sensors(self, accel=None, gyro=None, magn=None, baro=None):
        df_merged = None

        if accel is not None:
            df_accel = pd.DataFrame(accel, columns=['A_x', 'A_y', 'A_z'])
            df_merged = df_accel
        if gyro is not None:
            df_gyro = pd.DataFrame(gyro, columns=['G_x', 'G_y', 'G_z'])
            if df_merged is not None:
                df_merged = pd.concat([df_merged, df_gyro], axis=1)
            else:
                df_merged = df_gyro
        if magn is not None:
            df_magn = pd.DataFrame(magn, columns=['M_x', 'M_y', 'M_z'])
            if df_merged is not None:
                df_merged = pd.concat([df_merged, df_magn], axis=1)
            else:
                df_merged = df_magn
        if baro is not None:
            df_baro = pd.DataFrame(baro, columns=['B_p', 'B_t'])
            if df_merged is not None:
                df_merged = pd.concat([df_merged, df_baro], axis=1)
            else:
                df_merged = df_baro

        return df_merged

if __name__ == '__main__':
    dataset = FallAllDDataset("B:\\Dataset\\FallAllD\\", accellerometer=True, gyroscope=False, magnetometer=False, barometer=False)

    sample = dataset[3]

    print("Dataset classes: ", dataset.classes())
    print(sample['A'].shape)
    dataset.setVisualization(False)

    features, label = dataset[4]
    print(f"Label: {label}")
    print(f"Features shape: {features.shape}")
    print("Window classes: ", dataset.classes())

