import numpy as np
import pandas as pd
import os
import glob
import warnings
import re
from scipy.signal import butter, filtfilt
import scipy.ndimage
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ================= 1. 全局参数 =================
RADAR_FS = 20.0
GT_FS = 125.0
FFT_SIZE = 256
DATA_ROOT = r"D:\Desktop\FMCW radar-based multi-person vital sign monitoring data"
OUT_DIR = os.path.join(DATA_ROOT, "Mamba_Dataset_Complex")

if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
BIN_RES = 0.05 * 200 / FFT_SIZE

POSITION_PRIORS = {
    1: {"ang1": -40.0, "ang2": 20.0, "r1": 1.0, "r2": 1.0},
    2: {"ang1": -20.0, "ang2": 40.0, "r1": 1.4, "r2": 1.4},
    3: {"ang1": -40.0, "ang2": 40.0, "r1": 1.4, "r2": 1.0},
    4: {"ang1": -20.0, "ang2": 20.0, "r1": 1.0, "r2": 0.6},
    5: {"ang1": -20.0, "ang2": 40.0, "r1": 1.4, "r2": 1.0},
    6: {"ang1": -40.0, "ang2": 20.0, "r1": 0.6, "r2": 1.0},
    7: {"ang1": -40.0, "ang2": 40.0, "r1": 0.6, "r2": 0.6},
    8: {"ang1": -40.0, "ang2": 40.0, "r1": 1.0, "r2": 1.0},
    9: {"ang1": -20.0, "ang2": 20.0, "r1": 1.4, "r2": 1.4},
}

def load_dca1000_mimo_8rx(bin_path):
    raw = np.fromfile(bin_path, dtype=np.int16).astype(np.float32)
    comp = np.zeros(len(raw) // 2, dtype=complex)
    comp[0::2] = raw[0::4] + 1j * raw[2::4]
    comp[1::2] = raw[1::4] + 1j * raw[3::4]
    num_samples = 200; num_rx = 4; num_tx = 3
    total_chirps = len(comp) // (num_samples * num_rx)
    comp = comp[:total_chirps * num_samples * num_rx]
    cube = comp.reshape(total_chirps, num_samples, num_rx).transpose(0, 2, 1)
    frames = total_chirps // num_tx
    cube = cube[:frames * num_tx].reshape(frames, num_tx, num_rx, num_samples)
    cube_8rx = np.concatenate([cube[:, 0, :, :], cube[:, 2, :, :]], axis=1)
    return cube_8rx.transpose(0, 2, 1)

def extract_zf_signal(rp, bin_idx, target_angle, null_angle):
    n_frames, n_bins, n_rx = rp.shape
    rd = rp[:, bin_idx, :]
    d = 0.5
    phi_t = 2 * np.pi * d * np.sin(np.deg2rad(target_angle))
    a_t = np.exp(-1j * np.arange(n_rx) * phi_t).reshape(-1, 1)
    phi_n = 2 * np.pi * d * np.sin(np.deg2rad(null_angle))
    a_n = np.exp(-1j * np.arange(n_rx) * phi_n).reshape(-1, 1)

    C = np.hstack([a_t, a_n])
    g = np.array([[1], [0]])
    try:
        C_H_C_inv = np.linalg.inv(np.dot(C.conj().T, C))
        w = np.dot(np.dot(C, C_H_C_inv), g)
        sig = np.dot(rd, w.conj()).flatten()
    except np.linalg.LinAlgError:
        w = a_t / np.linalg.norm(a_t) ** 2
        sig = np.dot(rd, w.conj()).flatten()
    return sig

def extract_ground_truth_fft(csv_path, expected_len_radar):
    try:
        df_raw = pd.read_csv(csv_path, header=None)
        ecg_col_idx = 0
        for col in range(df_raw.shape[1]):
            if any('ecg' in str(val).lower() for val in df_raw.iloc[0:3, col]):
                ecg_col_idx = col
                break
        ecg_data = pd.to_numeric(df_raw.iloc[2:, ecg_col_idx], errors='coerce').dropna().values
        nyq = 0.5 * GT_FS
        b, a = butter(4, [0.9 / nyq, 2.0 / nyq], btype='band')
        clean_ecg = filtfilt(b, a, ecg_data)

        window_pts = int(8.0 * GT_FS); step_pts = int(0.5 * GT_FS)
        bpm_list, time_list = [], []

        if len(clean_ecg) < window_pts: return np.full(expected_len_radar, 75.0)

        for start in range(0, len(clean_ecg) - window_pts + 1, step_pts):
            segment = clean_ecg[start:start + window_pts]
            n_fft = len(segment) * 10
            freqs = np.fft.rfftfreq(n_fft, d=1 / GT_FS)
            fft_mag = np.abs(np.fft.rfft(segment, n=n_fft))
            valid_idx = np.where((freqs >= 0.9) & (freqs <= 2.0))[0]
            if len(valid_idx) == 0:
                bpm_list.append(75.0)
                continue
            dominant_f = freqs[valid_idx[np.argmax(fft_mag[valid_idx])]]
            bpm_list.append(dominant_f * 60.0)
            time_list.append((start + window_pts / 2.0) / GT_FS)

        if not bpm_list: return np.full(expected_len_radar, 75.0)
        radar_duration = expected_len_radar / RADAR_FS
        radar_time_axis = np.linspace(0, radar_duration, expected_len_radar)
        aligned_bpm = np.interp(radar_time_axis, time_list, bpm_list, left=bpm_list[0], right=bpm_list[-1])
        aligned_bpm = scipy.ndimage.gaussian_filter1d(aligned_bpm, sigma=RADAR_FS * 1.5)
        return np.clip(aligned_bpm, 54.0, 120.0)
    except: return np.full(expected_len_radar, 75.0)

def build_ecg_registry():
    registry = {}
    for csv_path in glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True):
        c_name = os.path.basename(csv_path).lower()
        t_match, f_match, p_match, i_match = re.search(r'target\s*(\d)', c_name), re.search(r'(2_5ghz|2ghz|3ghz)', c_name), re.search(r'positi?on_?\s*(\d)', c_name), re.search(r'\(\s*(\d+)\s*\)', c_name)
        if t_match and f_match and p_match and i_match:
            registry[(int(t_match.group(1)), f_match.group(1), int(p_match.group(1)), int(i_match.group(1)))] = csv_path
    return registry

def generate_dataset():
    ecg_registry = build_ecg_registry()
    bin_files = glob.glob(os.path.join(DATA_ROOT, "**", "adc_*.bin"), recursive=True)
    X_data_temp, Y_data_temp, P_data_temp = [], [], [] # 🌟 新增保存位置信息的数组

    for idx, bin_path in enumerate(bin_files):
        filename = os.path.basename(bin_path).lower()
        f_match, p_match, i_match = re.search(r'(2_5ghz|2ghz|3ghz)', filename), re.search(r'positi?on_?\s*(\d)', filename), re.search(r'\(\s*(\d+)\s*\)', filename)
        if not (f_match and p_match and i_match): continue

        f_id, p_id, i_id = f_match.group(1), int(p_match.group(1)), int(i_match.group(1))
        if p_id not in POSITION_PRIORS: continue
        params = POSITION_PRIORS[p_id]

        csv1_path, csv2_path = ecg_registry.get((1, f_id, p_id, i_id)), ecg_registry.get((2, f_id, p_id, i_id))
        if not csv1_path or not csv2_path: continue

        print(f"提取 Pos{p_id} | {f_id} | 序号({i_id}) ...")
        try:
            cube_8rx = load_dca1000_mimo_8rx(bin_path)
            alpha = 0.99
            mti_cube = np.zeros_like(cube_8rx)
            bg_acc = cube_8rx[0].copy()
            for t in range(cube_8rx.shape[0]):
                bg_acc = alpha * bg_acc + (1 - alpha) * cube_8rx[t]
                mti_cube[t] = cube_8rx[t] - bg_acc
            rp = np.fft.fft(mti_cube * np.hanning(mti_cube.shape[1]).reshape(1, -1, 1), n=FFT_SIZE, axis=1)

            bin_t1, bin_t2 = int((params['r1'] - 0.15) / BIN_RES), int((params['r2'] - 0.15) / BIN_RES)
            c1 = extract_zf_signal(rp, bin_t1, params['ang1'], params['ang2'])
            c2 = extract_zf_signal(rp, bin_t2, params['ang2'], params['ang1'])

            bpm1, bpm2 = extract_ground_truth_fft(csv1_path, len(c1)), extract_ground_truth_fft(csv2_path, len(c2))

            # 🌟 每次成对保存特征和标签的同时，保存当前的 position_id
            X_data_temp.extend([c1.astype(np.complex64), c2.astype(np.complex64)])
            Y_data_temp.extend([bpm1.astype(np.float32), bpm2.astype(np.float32)])
            P_data_temp.append(p_id)
        except: continue

    if not X_data_temp: return
    TARGET_LEN = min([len(x) for x in X_data_temp])
    TARGET_LEN = TARGET_LEN if TARGET_LEN > 1000 else 1150

    X_arr = np.array([x[:TARGET_LEN] for x in X_data_temp])
    Y_arr = np.array([y[:TARGET_LEN] for y in Y_data_temp])
    P_arr = np.array(P_data_temp) # 形状: (文件对数,)

    np.save(os.path.join(OUT_DIR, "radar_X_features.npy"), X_arr)
    np.save(os.path.join(OUT_DIR, "ecg_Y_labels.npy"), Y_arr)
    np.save(os.path.join(OUT_DIR, "radar_P_positions.npy"), P_arr) # 🌟 核心：保存位置映射表
    print(f"\n✅ 数据集已更新！包含特征: {X_arr.shape}, 位置标签: {P_arr.shape}")

if __name__ == "__main__":
    generate_dataset()