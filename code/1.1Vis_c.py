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
        Y_NO = 1 - Y_II - Y_NPQ
        NPQ_4 = (NPQ)/4
        FV_FM = (FM - F0) / FM
        y_valid_indices = np.where((Y_II > 0) & (Y_II < 1))

        y_labels = Y_II[y_valid_indices]

        x_features = x_bands[:, y_valid_indices[0], y_valid_indices[1]].T
        return x_features, y_labels

def apply_sg_smoothing(x_features, window_length=5, polyorder=2):
    smoothed_features = np.zeros_like(x_features)
    for i in range(x_features.shape[1]):
        smoothed_features[:, i] = savgol_filter(x_features[:, i], window_length, polyorder)
    return smoothed_features


def calculate_correlations(x_features, y_labels):
    num_samples, num_scales, num_features_per_scale = x_features.shape

    # A matrix used to store correlations between features and labels at each scale
    corr_matrix = np.zeros((num_scales, num_features_per_scale))

    # For each scale and each feature, the correlation is calculated
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
        all_x_features.append(x_features)
        all_y_labels.append(y_labels)
    else:
        print(f"No matching file for {x_path}")

all_x_features = np.concatenate(all_x_features, axis=0)
all_y_labels = np.concatenate(all_y_labels, axis=0)
all_x_features = all_x_features.astype('float32')
all_y_labels = all_y_labels.astype('float32')
b = all_x_features
z_scores = np.abs(zscore(all_x_features, axis=0))
b = all_x_features[(z_scores < 3).all(axis=1)]
all_y_labels = all_y_labels[(z_scores < 3).all(axis=1)]

print(f"all_x_features shape after removing outliers: {all_x_features.shape}")
print(f"all_y_labels shape after removing outliers: {all_y_labels.shape}")

#chl
NDVI = (b[:,82]-b[:,58])/(b[:,82]+b[:,58])
RR = 1/b[:, 39]
mSR = (b[:, 64]-b[:, 14])/(b[:, 64]+b[:, 14])
PSSRa = b[:, 82]/b[:, 59]#band800/band675
PSSRb = b[:, 82]/b[:, 54]#band800/band650
RARSa = b[:, 59]/b[:, 63]#band675/band700
mNDVI = (b[:, 73]-b[:, 64])/(b[:, 73]+b[:, 64])#(band750-band705)/(band750+band705)
mNDI = (b[:, 73]-b[:, 64])/(b[:, 73]-b[:, 64]-2*b[:, 14])#(band750-band705)/(band750-band705-2*band445)
CI = b[:, 73]/b[:, 65] #band750/band710
SRPI = b[:, 11]/b[:, 60]#band430/band6801
NPCI = (b[:, 60]-b[:, 11])/(b[:, 60]+b[:, 11])#(band680-band430)/(band680+band430)
CTRI1 = b[:, 62]/b[:, 10] #band695/band420
CAR = b[:, 62]/b[:, 75] #band695/band760

RI708 = b[:, 65]/b[:, 78]#band708/band775
ND780 = (b[:, 79]-b[:, 66])/(b[:, 79]+b[:, 66])#(band780-band712)/(band780+band712)
CCI = (b[:, 31]-b[:, 53])/(b[:, 31]+b[:, 53])#(band531-band645)(band531+band645)

#Car
RARSC = b[:, 75]/b[:, 25]#band760/band500
PSSRC = b[:, 82]/b[:, 19]#band800/band470
PRI = (b[:, 31]-b[:, 39])/(b[:, 31]+b[:, 39])#(band531-band570)/(band531+band570)
CRI550 = (1/b[:, 27])-(1/b[:, 35])#(1/band510)-(1/band550)
cri700 = (1/b[:, 27])-(1/b[:, 63])#(1/band510)-(1/band700)
CRI515 = (1/b[:, 28])-(1/b[:, 63])#(1/band515)-(1/band550)
CRI515_700 = (1/b[:, 28])-(1/b[:, 63])#(1/band515)-(1/band700)
RI531 = b[:, 31]/b[:, 82]#band530/band800

#ant
ARI = (1/b[:, 35])-(1/b[:, 63])#1/band550- 1/band700
PRI515 = (b[:, 28]-b[:, 31])/(b[:, 28]+b[:, 31])#(band515-band531)/(band515+band531)
PRIm1 = (b[:, 27]-b[:, 31])/(b[:, 27]+b[:, 31])#(band512-band531)/(band512+band531)
PRIm2 = (b[:, 44]-b[:, 31])/(b[:, 44]+b[:, 31])#(band600-band531)/(band600+band531)
PRIm3 = (b[:, 58]-b[:, 31])/(b[:, 58]+b[:, 31])#(band670-band531)/(band670+band531)
PRIm4 = (b[:, 39]-b[:, 31]-b[:, 58])/(b[:, 39]+b[:, 31]-b[:, 58])#(band570-band531-band670)/(band570+band531+band670)
PRI_ci = (b[:, 39]-b[:, 30])/(b[:, 39]+b[:, 31])*((b[:, 75]/b[:, 63])-1)#(band570-band530)/(band570+band530)*((band760/band700)-1)
PRIn = ((b[:, 39]-b[:, 31])/(b[:, 39]+b[:, 31]))/((b[:, 82]-b[:, 58])/np.sqrt(b[:, 82]+b[:, 58])*(b[:, 63]/b[:, 58]))
#Water
wi = (b[:, 101])-(b[:, 114])#band900/band970

#plant stress
HI_2013= (b[:, 31 ]-b[:, 63])/(b[:, 31]+b[:, 63])-0.5*b[:, 64]#(band534 -band698)/(band534+band698)-0.5*band704
HI_2014 = (b[:, 71]-b[:, 5])/(b[:, 71]+b[:, 5])-0.5*b[:, 5]#(band739 -band402)/(band739+band402)-0.5*band403
PSRI = (b[:, 60]-b[:, 25])/b[:, 73]#(band680-band500)/band750
NPQI = (b[:, 8]-b[:, 12])/(b[:, 8]+b[:, 12])#(band415 - band435)/(band415+band435)
#Chlorophyll fluorescence
CUR = (b[:, 59]*b[:, 62])/(b[:, 60]*b[:, 60])#(band675*band690)/band698*band698

print(f"CUR: {CUR.shape}")

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
#Chlorophyll fluorescence
     #add all other indices here in the same manner
    #'Y': all_y_labels # This is your response variable
}


# Insert the missing value
#print(f"all_x_features type: {type(indices_data)}, shape: {indices_data.shape}")
#print(f"all_y_labels type: {type(all_y_labels)}, shape: {all_y_labels.shape}")

# Make sure the data format is correct
indices_data_df = pd.DataFrame(indices_data)
all_y_labels_df = pd.DataFrame(all_y_labels)

print(f"all_x_features_df shape: {indices_data_df.shape}")
print(f"all_y_labels_df shape: {all_y_labels_df.shape}")

# Save the features and labels to an HDF5 file
with pd.HDFStore('Y_II_vis_lianxu_zhengti.h5', mode='w') as store:
    store.put('features', indices_data_df, format='table')
    store.put('y', all_y_labels_df, format='table')

