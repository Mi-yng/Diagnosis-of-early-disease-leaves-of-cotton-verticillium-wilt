import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
def resample_raster(source_path, target_path, out_path):
    with rasterio.open(target_path) as target:
        target_width, target_height = target.width, target.height

    with rasterio.open(source_path) as source:
        # 获取源文件的元数据
        kwargs = source.meta.copy()

        # 更新元数据为目标图像的尺寸
        kwargs.update({
            "width": target_width,
            "height": target_height,
            "transform": rasterio.Affine(source.transform.a, source.transform.b, source.transform.c,
                                         source.transform.d, source.transform.e, source.transform.f)
        })

        with rasterio.open(out_path, 'w', **kwargs) as dest:
            for i in range(1, source.count + 1):
                # 对每个波段进行重采样，这里使用最近邻重采样方法
                resampled_array = source.read(
                    i,
                    out_shape=(target_height, target_width),
                    resampling=Resampling.nearest
                )

                dest.write(resampled_array, i)

# 调用函数进行重采样
source_tiff_path = 'E:/2023年成像光谱/0603——beij/b1-7.tif'  # 要重采样的多波段图像路径
target_tiff_path = 'E:/2023FLUENCEtiff/0602_xiaochubeijing/b1-7.tif'  # 目标图像路径，用于匹配像素大小
resampled_image_path = 'E:/test/resampled_b1-7f.tif'  # 输出的重采样图像路径

resample_raster(source_tiff_path, target_tiff_path, resampled_image_path)

image_path = "E:/2023FLUENCEtiff/0602_xiaochubeijing/b1-7.tif"
with rasterio.open(image_path) as src:
     multiband_image = src.read()  # multiband_image的维度将为(波段数, 高, 宽)

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

image_path2 = 'E:/test/resampled_b1-7f.tif'
with rasterio.open(image_path2) as src:
    multiband_image = src.read()
band1 = multiband_image[40, :, :]#注意波段1为[0,:,:]
band2 = multiband_image[32, :, :]
PRI = (band2 - band1) / (band2 + band1)

fig, axs = plt.subplots(1, 2, figsize=(12,6),)
# 第一个子图显示 NDVI
im0 = axs[0].imshow(Y_NPQ, cmap='rainbow',vmin=0.05,vmax=0.32)
axs[0].set_title('Y_NPQ', fontsize=18)
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
#axs[0].axis('off')  # 关闭坐标轴

im2 = axs[1].imshow(PRI, cmap='rainbow',vmin=0.02,vmax=0.32)
axs[1].set_title('PRI',fontsize=18)
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
#axs[2].axis('off')
plt.savefig('bijiao2.jpg', bbox_inches='tight', dpi=300)
plt.show()