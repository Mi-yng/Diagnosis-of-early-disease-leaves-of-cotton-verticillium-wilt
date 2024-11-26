import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import median_filter

def preprocess_image_only_inside_leaf(image, leaf_mask):
    # 应用中值滤波去除异常值
    filtered_image = median_filter(image, size=6)

    # 找到缺失值且在叶片内部的位置
    missing_value_mask = np.isnan(filtered_image) & (leaf_mask == 255)

    # 只对叶片内部的缺失值进行插值填补
    filtered_image[missing_value_mask] = np.interp(
        np.flatnonzero(missing_value_mask),  # 缺失值的索引
        np.flatnonzero(~missing_value_mask),  # 非缺失值的索引
        filtered_image[~missing_value_mask])  # 非缺失值用于插值

    return filtered_image


def erode_and_extract_contours(image, num_erosions, erosion_size, erosion_iterations=1):
    contours_list = []
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded = image.copy()

    for i in range(num_erosions):
        # 腐蚀操作，每次迭代的次数相同
        eroded = cv2.erode(eroded, kernel, iterations=erosion_iterations)

        # 使用 Canny 边缘检测提取新的边缘
        edges = cv2.Canny(eroded, 100, 200)

        # 提取边缘对应的轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)


    return contours_list

def extract_layer_pixels(image, contours, layer_index):
    """根据给定的轮廓提取特定层的像素值。"""
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [contours[layer_index]], -1, 255, -1)
    return image[mask == 255]
def calculate_pixel_ratio(preprocessed_Y_NPQ, threshold):
    #ratios = []

        layer_pixels = preprocessed_Y_NPQ
        high_value_pixels = layer_pixels[layer_pixels > threshold]
        ratio = len(high_value_pixels) / len(layer_pixels) if len(layer_pixels) > 0 else 0
        #ratios.append(ratio)
        return ratio

def calculate_pixel_ratio2(preprocessed_Y_NPQ, threshold):
    #ratios = []

        layer_pixels = preprocessed_Y_NPQ
        low_value_pixels = layer_pixels[layer_pixels < threshold]
        ratio = len(low_value_pixels) / len(layer_pixels) if len(layer_pixels) > 0 else 0
        #ratios.append(ratio)
        return ratio

def process_image(file_path):
    """读取并处理图像，计算每层的荧光参数并收集统计数据。"""
    with rasterio.open(file_path) as src:
        image_data = src.read()
        image_data = np.where(image_data == src.nodata, np.nan, image_data)
        leaf_mask = np.where(image_data[1, :, :] > 0, 255, 0).astype(np.uint8)
        FM = preprocess_image_only_inside_leaf(image_data[1, :, :],leaf_mask)
        FM_ = preprocess_image_only_inside_leaf(image_data[35, :, :],leaf_mask)
        F0 = preprocess_image_only_inside_leaf(image_data[0, :, :],leaf_mask)
        F20 = preprocess_image_only_inside_leaf(image_data[34, :, :],leaf_mask)

    mask = np.where(FM > 0, 255, 0).astype(np.uint8)
    I_NPQ = (FM - FM_) / FM_
    I_F0_ = F0 / (((FM - F0) / FM) + F0 / FM)
    I_qp = (FM_ - F20) / (FM_ - F0)
    I_ql = I_qp * (I_F0_ / F20)
    I_Y_II = (FM_ - F20) / FM_
    I_Y_NPQ = 1 - I_Y_II - 1 / (I_NPQ + 1 + I_ql * (FM / F0 - 1))
    I_NPQ_4 = I_NPQ / 4
    I_Y_NO = 1 / (I_NPQ + 1 + I_ql * (FM / F0))
    I_FV_FM = (FM - F0) / FM
    I_qN = (FM - FM_) / (FM - F20)
    contours_list = erode_and_extract_contours(mask, 10, 5, 4)

    results = []
    for i, contours in enumerate(contours_list):
        layer_mask = np.zeros_like(FM, dtype=np.uint8)
        cv2.drawContours(layer_mask, contours, -1, 255, -1)
        masked_FM = FM[layer_mask == 255]
        masked_FM_ = FM_[layer_mask == 255]
        masked_F0 = F0[layer_mask == 255]
        masked_F20 = F20[layer_mask == 255]

        # Calculate fluorescence parameters for this layer
        NPQ = (masked_FM - masked_FM_) / masked_FM_
        F0_ = masked_F0 / (((masked_FM - masked_F0) / masked_FM) + masked_F0 / masked_FM)
        qp = (masked_FM_ - masked_F20) / (masked_FM_ - F0_)
        ql = qp * (F0_ / masked_F20)
        Y_II = (masked_FM_ - masked_F20) / masked_FM_
        Y_NPQ = 1 - Y_II - 1 / (NPQ + 1 + ql * (masked_FM / masked_F0 - 1))
        NPQ_4 = NPQ/4
        Y_NO = 1 /(NPQ + 1 +ql * (masked_FM / masked_F0))
        FV_FM = (masked_FM - masked_F0)/masked_FM
        qN = (masked_FM - masked_FM_)/(masked_FM - masked_F20)
        # Calculate statistical data

        median_NPQ_4 = np.nanmedian(NPQ_4)
        mean_NPQ_4 = np.nanmean(NPQ_4)
        std_NPQ_4 = np.nanstd(NPQ_4)
        pixel_ratio_NPQ_4 = calculate_pixel_ratio(NPQ_4, 0.24)

        median_Y_NO = np.nanmedian(Y_NO)
        mean_Y_NO = np.nanmean(Y_NO)
        std_Y_NO = np.nanstd(Y_NO)
        pixel_ratio_Y_NO = calculate_pixel_ratio(Y_NO, 0.31)

        median_FV_FM = np.nanmedian(FV_FM)
        mean_FV_FM = np.nanmean(FV_FM)
        std_FV_FM= np.nanstd(FV_FM)
        pixel_ratio_FV_FM = calculate_pixel_ratio2(FV_FM, 0.73)

        median_Y_II = np.nanmedian(Y_II)
        mean_Y_II = np.nanmean(Y_II)
        std_Y_II = np.nanstd(Y_II)
        pixel_ratio_Y_II = calculate_pixel_ratio2(Y_II,  0.48)

        median_ql = np.nanmedian(ql)
        mean_ql = np.nanmean(ql)
        std_ql = np.nanstd(ql)
        pixel_ratio_ql = calculate_pixel_ratio2(ql,  0.40)

        median_Y_NPQ = np.nanmedian(Y_NPQ)
        mean_Y_NPQ = np.nanmean(Y_NPQ)
        std_Y_NPQ = np.nanstd(Y_NPQ)
        pixel_ratio_Y_NPQ = calculate_pixel_ratio(Y_NPQ,  0.25)
        layer_stats = {
            'Layer': i + 1,
            'Median_NPQ_4': median_NPQ_4,
            'Mean_NPQ_4': mean_NPQ_4,
            'Std_NPQ_4': std_NPQ_4,
            'pixel_ratio_NPQ_4': pixel_ratio_NPQ_4,

            'Median_Y_NO': median_Y_NO,
            'Mean_Y_NO': mean_Y_NO,
            'Std_Y_NO': std_Y_NO,
            'pixel_ratio_Y_NO': pixel_ratio_Y_NO,

            'Median_FV_FM': median_FV_FM,
            'Mean_FV_FM': mean_FV_FM,
            'Std_FV_FM': std_FV_FM,
            'pixel_ratio_FV_FM': pixel_ratio_FV_FM,

            'Median_Y_II': median_Y_II,
            'Mean_Y_II': mean_Y_II,
            'Std_Y_II': std_Y_II,
            'pixel_ratio_Y_II': pixel_ratio_Y_II,

            'Median_qN': median_ql,
            'Mean_qN': mean_ql,
            'Std_qN': std_ql,
            'pixel_ratio_qN': pixel_ratio_ql,

            'Median_Y_NPQ': median_Y_NPQ,
            'Mean_Y_NPQ': mean_Y_NPQ,
            'Std_Y_NPQ': std_Y_NPQ,
            'pixel_ratio_Y_NPQ':pixel_ratio_Y_NPQ

        }
        results.append(layer_stats)

    return results


def process_folder(folder_path):
    """处理文件夹中所有图像并收集数据。"""
    all_results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            image_results = process_image(file_path)
            for result in image_results:
                result['Image'] = filename
                all_results.append(result)
    return all_results

def save_results_to_csv(results, output_file):
    """将结果保存到CSV文件。"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)






# 使用
folder_path = "E:/2023FLUENCEtiff/low_deal"
output_file = 'fluorescence_layers_low_deal_total.csv'


results = process_folder(folder_path)
save_results_to_csv(results, output_file)
