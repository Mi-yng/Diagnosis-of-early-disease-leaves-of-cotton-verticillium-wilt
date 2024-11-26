import glob
import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import KBinsDiscretizer
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from matplotlib import font_manager
import pandas as pd


def get_matching_key(filename):
    return ''.join(filter(str.isdigit, os.path.basename(filename)))


def process_image(x_path, y_path):
    with rasterio.open(x_path) as x_src, rasterio.open(y_path) as y_src:
        width = x_src.width // 2
        height = x_src.height // 2

        x_bands = x_src.read(out_shape=(x_src.count, height, width), resampling=Resampling.bilinear)
        y_band = y_src.read(out_shape=(y_src.count, height, width), resampling=Resampling.bilinear)

        FM = y_band[1, :, :]
        FM_ = y_band[39, :, :]
        F0 = y_band[0, :, :]
        F20 = y_band[38, :, :]
        NPQ = (FM - FM_) / FM_
        F0_ = F0 / (((FM - F0) / FM) + F0 / FM_)
        qp = (FM_ - F20) / (FM_ - F0_)
        ql = qp * (F0_ / F20)
        Y_II = (FM_ - F20) / FM_
        Y_NPQ = 1 - Y_II - 1 / (NPQ + 1 + ql * (FM / F0 - 1))
        FV_FM = (FM - F0) / FM
        Y_NO = 1 - Y_II - Y_NPQ
        NPQ_4 = (NPQ) / 4
        y_valid_indices = np.where((Y_II > 0) &( Y_II < 1))

        y_labels = Y_II[y_valid_indices]

        x_features = x_bands[:, y_valid_indices[0], y_valid_indices[1]].T
        return x_features, y_labels


def mexican_hat_wavelet(length, scale):
    sigma = scale / np.sqrt(2)
    t = np.linspace(-length / 2, length / 2, length)
    wavelet = (2 / (np.sqrt(3 * sigma) * np.pi ** 0.25)) * (1 - (t / sigma) ** 2) * np.exp(-(t ** 2) / (2 * sigma ** 2))
    return wavelet


def mexican_hat_wavelet_transform(data, scales):
    coefficient_matrix = []
    for scale in scales:
        wavelet = mexican_hat_wavelet(len(data), scale)
        coefficients = signal.convolve(data, wavelet, mode='same')
        coefficient_matrix.append(coefficients)
        # 转换成NumPy数组并返回
    coefficient_matrix = np.array(coefficient_matrix)
    return np.array(coefficient_matrix)


def apply_sg_smoothing(x_features, window_length=5, polyorder=2):
    smoothed_features = np.zeros_like(x_features)
    for i in range(x_features.shape[1]):
        smoothed_features[:, i] = savgol_filter(x_features[:, i], window_length, polyorder)
    return smoothed_features


def calculate_correlations(x_features, y_labels):
    num_samples, num_scales, num_features_per_scale = x_features.shape

    # 用于存储每个尺度下特征与标签之间的相关性的矩阵
    corr_matrix = np.zeros((num_scales, num_features_per_scale))

    # 对于每个尺度和每个特征，计算相关性
    for scale_idx in range(num_scales):
        for feature_idx in range(num_features_per_scale):
            # 提取当前尺度当前特征的值
            feature_values = x_features[:, scale_idx, feature_idx]
            # 计算特征值和标签值之间的相关系数
            correlation = np.corrcoef(feature_values, y_labels)[0, 1]
            corr_matrix[scale_idx, feature_idx] = correlation

    return corr_matrix





x_folder = 'E:/2023FLUENCEtiff/test_xx/'
y_folder = 'E:/2023FLUENCEtiff/test_yy/'
x_filepaths = glob.glob(os.path.join(x_folder, '*.tif'))
y_filepaths = glob.glob(os.path.join(y_folder, '*.tif'))
x_files_mapping = {get_matching_key(filepath): filepath for filepath in x_filepaths}
y_files_mapping = {get_matching_key(filepath): filepath for filepath in y_filepaths}

all_x_features = []
all_y_labels = []

for key in x_files_mapping.keys():
    x_path = x_files_mapping[key]
    y_path = y_files_mapping.get(key)
    if y_path:
        x_features, y_labels = process_image(x_path, y_path)
        x_features_smoothed = apply_sg_smoothing(x_features)
        #x_features_continuum_removed = np.array([continuum_removal(feature) for feature in x_features_smoothed.T]).T
        all_x_features.append(x_features_smoothed)
        all_y_labels.append(y_labels)
    else:
        print(f"No matching file for {x_path}")

all_x_features = np.concatenate(all_x_features, axis=0)
all_y_labels = np.concatenate(all_y_labels, axis=0)
all_x_features = all_x_features.astype('float32')
all_y_labels = all_y_labels.astype('float32')

z_scores = np.abs(zscore(all_x_features, axis=0))
all_x_features = all_x_features[(z_scores < 3).all(axis=1)]
all_y_labels = all_y_labels[(z_scores < 3).all(axis=1)]

print(f"all_x_features shape after removing outliers: {all_x_features.shape}")
print(f"all_y_labels shape after removing outliers: {all_y_labels.shape}")

scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
batch_size = 7000

reshaped_data_list = []
for i in range(0, len(all_x_features), batch_size):
    batch_features = all_x_features[i:i + batch_size]
    wavelet_transformed_features = np.array([mexican_hat_wavelet_transform(x, scales) for x in batch_features])
    print(f"wavelet_transformed_features shape for batch {i // batch_size}: {wavelet_transformed_features.shape}")
    reshaped_data = wavelet_transformed_features.reshape(wavelet_transformed_features.shape[0], len(scales), -1)
    reshaped_data_list.append(reshaped_data)

reshaped_data = np.concatenate(reshaped_data_list, axis=0)

# 插补缺失值
imputer = SimpleImputer(strategy='mean')
reshaped_data = imputer.fit_transform(reshaped_data.reshape(reshaped_data.shape[0], -1))

print(f"reshaped_data shape after imputation: {reshaped_data.shape}")

# 缩放特征
#scaler = StandardScaler()
#reshaped_data = scaler.fit_transform(reshaped_data)

num_bands = reshaped_data.shape[1] // len(scales)

print(f"num_bands: {num_bands}")
print(f"reshaped_data.shape: {reshaped_data.shape}")

expected_size = len(all_x_features) * len(scales) * num_bands
actual_size = reshaped_data.size

# 打印调试信息
print(f"Expected size: {expected_size}")
print(f"Actual size: {actual_size}")
print(f"len(all_x_features): {len(all_x_features)}")
print(f"len(scales): {len(scales)}")
print(f"reshaped_data.shape[1]: {reshaped_data.shape[1]}")

# 计算并验证 reshaped_data 的尺寸是否正确
if expected_size != actual_size:
    raise ValueError(f"Mismatch in expected size {expected_size} and actual size {actual_size}")

# df = pd.DataFrame(all_x_features)
# df.to_hdf('waveleth.h5', key='indices', mode='w')
# 检查数据格式
print(f"all_x_features type: {type(all_x_features)}, shape: {all_x_features.shape}")
print(f"all_y_labels type: {type(all_y_labels)}, shape: {all_y_labels.shape}")

# 确保数据格式正确
all_x_features_df = pd.DataFrame(all_x_features)
all_y_labels_df = pd.DataFrame(all_y_labels)

print(f"all_x_features_df shape: {all_x_features_df.shape}")
print(f"all_y_labels_df shape: {all_y_labels_df.shape}")

reshaped_data_df = pd.DataFrame(reshaped_data)
# 检查 reshaped_data_df 是否符合预期
print(f"reshaped_data_df shape: {reshaped_data_df.shape}")
# 将特征和标签保存到 HDF5 文件中
with pd.HDFStore('Y_II_cwt_Lianxu_zhengti.h5', mode='w') as store:
    store.put('features', reshaped_data_df, format='table')
    store.put('labels', all_y_labels_df, format='table')