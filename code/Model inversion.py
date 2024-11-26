import glob
import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import joblib


def get_matching_key(filename):
    return ''.join(filter(str.isdigit, os.path.basename(filename)))

def mexican_hat_wavelet(length, scale):
    sigma = scale / np.sqrt(2)
    t = np.linspace(-length / 2, length / 2, length)
    wavelet = (2 / (np.sqrt(3 * sigma) * np.pi**0.25)) * (1 - (t / sigma)**2) * np.exp(-(t**2) / (2 * sigma**2))
    return wavelet

def mexican_hat_wavelet_transform(data, scales):
    coefficient_matrix = []
    for scale in scales:
        wavelet = mexican_hat_wavelet(len(data), scale)
        coefficients = signal.convolve(data, wavelet, mode='same')
        coefficient_matrix.append(coefficients)
    return np.array(coefficient_matrix)

def apply_sg_smoothing(x_features, window_length=5, polyorder=2):
    smoothed_features = np.zeros_like(x_features)
    for i in range(x_features.shape[1]):
        smoothed_features[:, i] = savgol_filter(x_features[:, i], window_length, polyorder)
    return smoothed_features

# 加载原始图像数据
with rasterio.open('E:/HSI-PEIDUI/re0608-b1-4.tif') as x_src:
    width = x_src.width
    height = x_src.height
    x_bands = x_src.read(out_shape=(x_src.count, height, width), resampling=Resampling.bilinear)
    print(f"Image width: {width}, Image height: {height}")

# 创建一个掩码，用于标记所有波段中为零的像素
mask = np.all(x_bands == 0, axis=0)  # 如果所有波段的值都为零，则将该像素标记为True

# 只对非零像素进行处理
non_zero_indices = np.where(~mask)

# 提取非零像素的光谱数据
b = x_bands[:, non_zero_indices[0], non_zero_indices[1]]

# chl - 计算所需指数
NDVI = (b[82]-b[58])/(b[82]+b[58])
RR = 1/b[39]
mSR = (b[64]-b[14])/(b[64]+b[14])
PSSRa = b[82]/b[59]
PSSRb = b[82]/b[54]
RARSa = b[59]/b[63]
mNDVI = (b[73]-b[64])/(b[73]+b[64])
mNDI = (b[73]-b[64])/(b[73]-b[64]-2*b[64])
CI = b[73]/b[65]
SRPI = b[11]/b[60]
NPCI = (b[60]-b[11])/(b[60]+b[11])
CTRI1 = b[63]/b[9]
CAR = b[62]/b[75]

RI708 = b[65]/b[78]
ND780 = (b[79]-b[66])/(b[79]+b[66])
CCI = (b[31]-b[53])/(b[31]+b[53])

# Car
RARSC = b[75]/b[25]
PSSRC = b[82]/b[19]
PRI = (b[31]-b[39])/(b[31]+b[39])
CRI550 = (1/b[27])-(1/b[35])
cri700 = (1/b[27])-(1/b[63])
CRI515 = (1/b[28])-(1/b[63])
CRI515_700 = (1/b[28])-(1/b[63])
RI531 = b[31]/b[82]

# ant
ARI = (1/b[35])-(1/b[63])
PRI515 = (b[28]-b[31])/(b[28]+b[31])
PRIm1 = (b[27]-b[31])/(b[27]+b[31])
PRIm2 = (b[44]-b[31])/(b[44]+b[31])
PRIm3 = (b[58]-b[31])/(b[58]+b[31])
PRIm4 = (b[39]-b[31]-b[58])/(b[39]+b[31]-b[58])
PRI_ci = (b[39]-b[30])/(b[39]+b[31])*((b[75]/b[63])-1)
PRIn = ((b[39]-b[31])/(b[39]+b[31]))/((b[82]-b[58])/np.sqrt(b[82]+b[58])*(b[63]/b[58]))

# Water
wi = (b[101])-(b[114])

# plant stress
HI_2013= (b[31 ]-b[63])/(b[31]+b[63])-0.5*b[64]
HI_2014 = (b[71]-b[5])/(b[71]+b[5])-0.5*b[5]
PSRI = (b[60]-b[25])/b[73]
NPQI = (b[8]-b[12])/(b[8]+b[12])

# Chlorophyll fluorescence
CUR = (b[59]*b[62])/(b[60]*b[60])

# 创建包含计算特征的 DataFrame
indices_data = {
    'NDVI': NDVI.flatten(),
    'RR': RR.flatten(),
    'mSR': mSR.flatten(),
    'PSSRa': PSSRa.flatten(),
    'PSSRb': PSSRb.flatten(),
    'RARSa': RARSa.flatten(),
    'mNDVI': mNDVI.flatten(),
    'mNDI': mNDI.flatten(),
    'CI': CI.flatten(),
    'SRPI': SRPI.flatten(),
    'NPCI': NPCI.flatten(),
    'CTRI1': CTRI1.flatten(),
    'CAR': CAR.flatten(),
    'RI708': RI708.flatten(),
    'ND780': ND780.flatten(),
    'CCI': CCI.flatten(),
    'RARSC': RARSC.flatten(),
    'PSSRC': PSSRC.flatten(),
    'PRI': PRI.flatten(),
    'CRI550': CRI550.flatten(),
    'cri700': cri700.flatten(),
    'CRI515': CRI515.flatten(),
    'PRIm1': PRIm1.flatten(),
    'PRIm2': PRIm2.flatten(),
    'PRIm3': PRIm3.flatten(),
    'PRIm4': PRIm4.flatten(),
    'PRI_ci': PRI_ci.flatten(),
    'PRIn': PRIn.flatten(),
    'wi': wi.flatten(),
    'HI_2013': HI_2013.flatten(),
    'HI_2014': HI_2014.flatten(),
    'PSRI': PSRI.flatten(),
    'NPQI': NPQI.flatten(),
    'CUR': CUR.flatten(),
}

indices_data_df = pd.DataFrame(indices_data)

# 计算小波变换特征
scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
reshaped_data = np.array([mexican_hat_wavelet_transform(band, scales) for band in
                          np.transpose(b, (1, 0))])
reshaped_data = reshaped_data.reshape(reshaped_data.shape[0], -1)
reshaped_data_df = pd.DataFrame(reshaped_data)

# 合并所有特征到一个 DataFrame 中
data = pd.concat([reshaped_data_df, indices_data_df], axis=1)
data.columns = range(data.shape[1])

data.columns = data.columns.astype(str)

# 处理无限大和 NaN 值，将其替换为合适的数值
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

# Step 1: 加载保存的 scaler
scaler = joblib.load('scaler_balanced_Y_II_2.pkl')

# Step 2: 对原始特征进行标准化
data_scaled = scaler.transform(data)

# Step 4: 加载保存的特征索引
with open('Y_II_selected_classification_features3.json', 'r') as f:
    c_selected_feature_indices = json.load(f)

with open('Y_II_selected_regression_features3.json', 'r') as f:
    r_selected_feature_indices = json.load(f)

# Step 4: 从标准化后的数据中选择特征
re_features = data_scaled[:, r_selected_feature_indices]  # 回归任务的特征
c_features = data_scaled[:, c_selected_feature_indices]   # 分类任务的特征

# Step 5: 加载训练好的多任务学习模型
model = load_model('Y_II_3_model.h5')

# Step 6: 先进行回归任务预测
_, regression_pred, precision_pred = model.predict([c_features, re_features])

# Step 7: 校正预测结果
# 假设0.48以上的回归预测被精准判别模型预测错误，我们将其设置为0.6
# 如果精准判别模型输出大于0.5，则说明这个区域需要调整为0.6
#corrected_pred = np.where((precision_pred < 0.4), 0.25, regression_pred)
#corrected_pred = np.where((regression_pred < 0.48) & (precision_pred < 0.48), 0.6, regression_pred)
#corrected_pred = np.where((precision_pred > 0.48),  # 如果精准判别模型输出大于 0.5
                          #np.where(regression_pred < 0.45, 0.3, 0.6),  # 回归预测 > 0.48 -> 0.6; 否则 -> 0.4
                          #regression_pred)  # 否则保持原回归预测值
corrected_pred = np.where(precision_pred < 0.4, 0.25,
                          np.where(precision_pred > 0.5, 0.6, regression_pred))
# Step 8: 重塑回图像的空间维度
# 我们从之前的 non_zero_indices 获取非零像素的位置，现在我们将预测值插回图像中
height, width = mask.shape  # 获取图像的高度和宽度
corrected_image = np.zeros((height, width))  # 创建一个全零的图像
r_image = np.zeros((height, width))
p_image = np.zeros((height, width))
# 将校正后的预测结果填充到非零像素中
corrected_image[non_zero_indices[0], non_zero_indices[1]] = corrected_pred.flatten()
r_image[non_zero_indices[0], non_zero_indices[1]] = regression_pred.flatten()
p_image[non_zero_indices[0], non_zero_indices[1]] = precision_pred.flatten()
# Step 9: 保存校正后的图像
with rasterio.open('corrected_prediction_B2-2.tif', 'w', driver='GTiff',
                   height=height, width=width, count=1, dtype=rasterio.float32) as dst:
    dst.write(corrected_image, 1)
with rasterio.open('r_image_B2-2.tif', 'w', driver='GTiff',
                   height=height, width=width, count=1, dtype=rasterio.float32) as dst:
    dst.write(r_image, 1)
with rasterio.open('p_image_B2-2.tif', 'w', driver='GTiff',
                    height=height, width=width, count=1, dtype=rasterio.float32) as dst:
    dst.write(p_image, 1)
print("Corrected prediction image saved as 'corrected_prediction_image.tif'")

# Step 1: 加载生成的预测图像
pred_image_path = 'corrected_prediction_B2-2.tif'
p_image_path = 'p_image_B2-2.tif'
r_image_path = 'r_image_B2-2.tif'

with rasterio.open(pred_image_path) as src:
    corrected_image = src.read(1)  # 读取图像的第一个波段
with rasterio.open(p_image_path) as src:
    p_image = src.read(1)  # 读取图像的第一个波段
with rasterio.open(r_image_path) as src:
    r_image = src.read(1)  # 读取图像的第一个波段
# Step 2: 可视化预测图像
plt.figure(figsize=(10, 8))
plt.imshow(corrected_image, cmap='rainbow', vmin=0.2, vmax=0.7)  # 使用 'viridis' 色彩映射
plt.colorbar(label='Predicted Values')  # 显示颜色条
plt.title('Corrected Prediction Image')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(r_image, cmap='rainbow', vmin=0.2, vmax=0.7)  # 使用 'viridis' 色彩映射
plt.colorbar(label='Predicted Values')  # 显示颜色条
plt.title('Corrected Prediction Image')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(p_image, cmap='rainbow', vmin=0.2, vmax=0.7)  # 使用 'viridis' 色彩映射
plt.colorbar(label='Predicted Values')  # 显示颜色条
plt.title('Corrected Prediction Image')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()