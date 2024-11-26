import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import font_manager
from scipy.ndimage import median_filter
def erode_and_extract_contours(image, num_erosions, erosion_size):
    contours_list = []
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded = image.copy()

    for i in range(1, num_erosions + 1):
        # 腐蚀操作
        eroded = cv2.erode(eroded, kernel, iterations=i)

        # 使用 Canny 边缘检测提取新的边缘
        edges = cv2.Canny(eroded, 100, 200)

        # 提取边缘对应的轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)

    return contours_list


def plot_contours_on_image(image, contours_list):
    # 为了可视化目的，创建一个用于绘制轮廓的彩色图像
    image = np.where(image > 0, 255, 0).astype(np.uint8)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 给每组轮廓分配不同的颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    # 开始绘制轮廓
    for i, contours in enumerate(contours_list):
        cv2.drawContours(color_image, contours, -1, colors[i % len(colors)], 2)

    plt.imshow(color_image)
    plt.axis('off')
    plt.show()

def preprocess_image_only_inside_leaf(image, leaf_mask):
    # 应用中值滤波去除异常值
    filtered_image = median_filter(image, size=3)

    # 找到缺失值且在叶片内部的位置
    missing_value_mask = np.isnan(filtered_image) & (leaf_mask == 255)

    # 只对叶片内部的缺失值进行插值填补
    filtered_image[missing_value_mask] = np.interp(
        np.flatnonzero(missing_value_mask),  # 缺失值的索引
        np.flatnonzero(~missing_value_mask),  # 非缺失值的索引
        filtered_image[~missing_value_mask])  # 非缺失值用于插值

    return filtered_image
def plot_contours_on_heatmap(heatmap, contours_list, contour_color='darkblue', line_style='solid'):
    plt.figure(figsize=(10, 8))
    # 显示热图
    h = plt.imshow(heatmap, cmap='PiYG_r', interpolation='nearest', vmin=0.08, vmax=0.35)
    #plt.colorbar(shrink=0.9)
    cb = plt.colorbar(h,shrink=0.6,orientation='horizontal')
    cb.ax.tick_params(labelsize=16)  # 设置色标刻度字体大小。
    cb.set_label('Y_NPQ', fontsize=16,fontweight='bold')  # 设置colorbar的标签字体及其大小
    my_font = font_manager.FontProperties(family='SimHei', size=16)
    #plt.subplot(1, 2, 1)
    #plt.plot(range(3, steps + 1), high_value_pixel_counts, marker='o', linewidth=2.0)
    #plt.xticks(fontsize=16, fontweight='bold')  # x轴刻度字体大小
    plt.yticks(fontsize=16, fontweight='bold')  # y轴刻度字体大小
    #plt.text(18, 550, '(a)', fontsize=16, fontweight='bold')
    #bwith = 2  # 边框宽度设置为2
    #TK = plt.gca()  # 获取边框
    #TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
    #TK.spines['left'].set_linewidth(bwith)  # 图框左边
    #TK.spines['top'].set_linewidth(bwith)  # 图框上边
    #TK.spines['right'].set_linewidth(bwith)  # 图框右边

    #plt.title('Y_NPQ高于0.18像素值', fontsize=16, fontweight='bold', fontproperties=my_font)
    #plt.xlabel('内缩层数', fontsize=16, fontweight='bold', fontproperties=my_font)
    plt.ylabel('高值像素', fontsize=16, fontweight='bold', fontproperties=my_font)
    # 在热图上绘制轮廓
    for contours in contours_list:
        for contour in contours:
            # 将轮廓坐标转换为(x, y)点对
            segments = np.reshape(contour, (-1, 2))
            # 创建LineCollection对象
            lc = LineCollection([segments], colors=contour_color, linestyle=line_style)
            # 将集合添加到当前轴
            plt.gca().add_collection(lc)

    plt.axis('off')  # 隐藏坐标轴
    plt.show()

# 从二值图像中读入叶子区域（即把叶子区域当作初始点）
image_path = "E:/2023FLUENCEtiff/0608_beijingxiaochu/08b1-4.tif"
with rasterio.open(image_path) as src:
     multiband_image = src.read()
     profile = src.profile# multiband_image的维度将为(波段数, 高, 宽)

FM = multiband_image[1, :, :]#注意波段1为[0,:,:]
FM_ = multiband_image[41, :, :]
F0 = multiband_image[0, :, :]
F20 = multiband_image[40, :, :]
NPQ = (FM - FM_) / (FM_)
NPQ_4 = NPQ/4
F0_ = F0 / (((FM - F0)/FM )+ F0 / FM_)
qp = (FM_ - F20) / (FM_ - F0_)
ql = qp*(F0_/F20)
Y_II = (FM_ - F20) / (FM_)
Y_NPQ = 1 - Y_II - 1 / (NPQ + 1 + ql*(FM/F0 - 1))
#Y_NPQ = np.where(Y_NPQ > 0.7, np.nan, Y_NPQ)
#Y_NPQ = np.where(Y_NPQ < 0, np.nan, Y_NPQ)
leaf_region_binary = np.where(Y_II > 0, 255, 0).astype(np.uint8)


num_erosions = 10 #内缩次数
erosion_size = 5 #内缩宽度

# 执行腐蚀和轮廓提取
contours_list = erode_and_extract_contours(leaf_region_binary, num_erosions, erosion_size)

# 将轮廓绘制到原始图像上
original_image = cv2.imread('path_to_original_image.jpg', cv2.IMREAD_GRAYSCALE)
plot_contours_on_image(Y_II, contours_list)
# 显示热图及边缘

plot_contours_on_heatmap(Y_II, contours_list, contour_color='darkblue',  line_style='solid')