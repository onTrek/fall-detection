import matplotlib.pyplot as plt
import numpy as np

def plot_sensor_data(data, sensor_type="accel", labels=None, vertical_accel=False, orientation=False):
    if sensor_type == "accel":
        expected_cols = ['accel_time_list', 'accel_x_list', 'accel_y_list', 'accel_z_list']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = data['accel_time_list']
        xs = data['accel_x_list']
        ys = data['accel_y_list']
        zs = data['accel_z_list']
    elif sensor_type == "gyro":
        expected_cols = ['gyro_time_list', 'gyro_x_list', 'gyro_y_list', 'gyro_z_list']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = data['gyro_time_list']
        xs = data['gyro_x_list']
        ys = data['gyro_y_list']
        zs = data['gyro_z_list']
    elif sensor_type == "orientation" and orientation:
        expected_cols = ['orientation_time_list','orientation_s_list','orientation_i_list','orientation_j_list','orientation_k_list']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = data['orientation_time_list']
        xs = data['orientation_s_list']
        ys = data['orientation_i_list']
        zs = data['orientation_j_list']
    elif sensor_type == "vertical_accel" and vertical_accel:
        expected_cols = ['vertical_Accel_x','vertical_Accel_y','vertical_Accel_z','accel_time_list']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = data['accel_time_list']
        xs = data['vertical_Accel_x']
        ys = data['vertical_Accel_y']
        zs = data['vertical_Accel_z']
    elif sensor_type == "A":
        expected_cols = ['x', 'y', 'z']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = np.arange(len(data['x']))
        xs = data['x']
        ys = data['y']
        zs = data['z']
    elif sensor_type == "G":
        expected_cols = ['x', 'y', 'z']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = np.arange(len(data['x']))
        xs = data['x']
        ys = data['y']
        zs = data['z']
    elif sensor_type == "M":
        expected_cols = ['x', 'y', 'z']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = np.arange(len(data['x']))
        xs = data['x']
        ys = data['y']
        zs = data['z']
    elif sensor_type == "B":
        expected_cols = ['pressure', 'temperature']
        if not all(col in data.columns for col in expected_cols):
            raise ValueError(f"Il DataFrame deve contenere le colonne: {expected_cols}")

        times = np.arange(len(data['pressure']))
        xs = data['pressure']
        ys = data['temperature']
        zs = np.zeros_like(xs)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, xs, label='X', color='red')
    plt.plot(times, ys, label='Y', color='green')
    plt.plot(times, zs, label='Z', color='blue')

    plt.title(labels)
    plt.xlabel('Tempo')
    plt.ylabel('Valore')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def kalman_filter_1d(data, process_variance=1e-5, measurement_variance=1e-2):
    n = len(data)
    x_est = np.zeros(n)
    P = np.zeros(n)
    Q = process_variance
    R = measurement_variance

    x_est[0] = data[0]
    P[0] = 1.0

    for k in range(1, n):

        x_pred = x_est[k-1]
        P_pred = P[k-1] + Q


        K = P_pred / (P_pred + R)
        x_est[k] = x_pred + K * (data[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x_est

def kalman(matrix):
    matrix = np.asarray(matrix)
    result = np.copy(matrix)

    # Se ci sono 4 colonne, salta la prima (timestamp)
    start_col = 1 if matrix.shape[1] == 4 else 0

    for col in range(start_col, matrix.shape[1]):
        result[:, col] = kalman_filter_1d(matrix[:, col])

    return result

def resample_sequence(sequence, original_hz=238, target_hz=20):
    sequence = np.asarray(sequence)
    step = int(round(original_hz / target_hz))
    downsampled = sequence[::step]
    return downsampled

def convert_accelerometer_for_android(raw_acc):
    ACC_LSB_TO_G = 0.000244  # Sensibilità ±8g
    acc_m_s2 = raw_acc * ACC_LSB_TO_G * 9.81
    return acc_m_s2

def convert_gyroscope_for_android(raw_gyro):
    GYRO_LSB_TO_DPS = 0.07  # Sensibilità ±2000 dps
    gyro_rad_s = raw_gyro * GYRO_LSB_TO_DPS * np.pi / 180
    return gyro_rad_s

def convert_magnetometer_for_android(raw_mag):
    MAG_LSB_TO_GAUSS = 0.00014  # ±4 Gauss
    mag_uT = raw_mag * MAG_LSB_TO_GAUSS * 100  # 1 Gauss = 100 µT
    return mag_uT


def convert_barometer_for_android(raw_barometer):
    barometer_array = raw_barometer[:, :2]
    return barometer_array