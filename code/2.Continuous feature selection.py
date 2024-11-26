
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
import json
import joblib  # 用于保存 scaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# 加载 HDF5 文件中的特征和标签
with pd.HDFStore('Y_NPQ_cwt_Lianxu_bupinghua_2.h5', mode='r') as store:
    x1 = store['features'].values
    y1 = store['labels'].values

with pd.HDFStore('Y_NPQ_vis_lianxu_bupinghua2.h5', mode='r') as store:
    x2 = store['features'].values
    y2 = store['y'].values

# 转换数据为 DataFrame 和 Series
x1_df = pd.DataFrame(x1)
x2_df = pd.DataFrame(x2)
y1_series = pd.Series(y1.flatten())
y2_series = pd.Series(y2.flatten())

# y1_series 的标签类型转换为分类数据类型
y1_series = y1_series.astype('category')

# 合并特征集
data = pd.concat([x1_df, x2_df], axis=1)
# 重新设置特征列的序号（0, 1, 2, ..., n）
data.columns = range(data.shape[1])
# 记录原始特征的索引
original_feature_indices = data.columns.tolist()
# 1. 检查无穷大和 NaN 值的数量
print("Number of infinity values:", np.isinf(data).sum().sum())
print("Number of NaN values:", np.isnan(data).sum().sum())

# 2. 替换无穷大和 NaN 值
# 将正无穷大替换为该特征的最大有限值，将负无穷大替换为最小有限值
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(data.mean(), inplace=True)

# 3. 确认处理后的无穷大和 NaN 值
print("After replacement - Number of NaN values:", np.isnan(data).sum().sum())


# 标准化特征
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=original_feature_indices)
joblib.dump(scaler, 'scaler_balanced_Y_NPQ.pkl')

# 特征选择函数 - 基于相关性
def select_features(data, y, threshold=0.2):
    correlations = data.apply(lambda x: x.corr(y))  # 计算相关系数
    selected_features = correlations[correlations.abs() > threshold].index  # 筛选相关系数大于阈值的特征索引
    return data[selected_features]  # 返回原始数据中对应的特征列

# 相关性筛选
data_selected = select_features(data, y2_series)

# 使用 Lasso 进行特征选择
lasso_cv = LassoCV(cv=5, random_state=42).fit(data_selected, y2_series)

# 打印最优的 alpha 参数
print(f"Optimal alpha: {lasso_cv.alpha_}")

# 使用最优的 alpha 训练 Lasso 模型
lasso = Lasso(alpha=lasso_cv.alpha_).fit(data_selected, y2_series)

# 获取特征的 Lasso 系数
lasso_coefficients = pd.Series(lasso.coef_, index=data_selected.columns)

# 按照系数的绝对值排序，选择前20个最重要的特征
top_20_features = lasso_coefficients.abs().sort_values(ascending=False).head(20).index

# 提取前20个特征
selected_data_top20 = data_selected[top_20_features]

# 打印被选中的前20个特征
print(f"Top 20 selected features: {top_20_features.tolist()}")

# 保存前20个特征的索引
selected_feature_top20_indices = [original_feature_indices.index(feature) for feature in top_20_features]

# 将原始特征索引保存到文件
with open('selected_feature_top20_YNPQ.json', 'w') as f:
    json.dump(selected_feature_top20_indices, f)

print("Top 20 feature indices saved to 'selected_feature_top20_YNPQ.json'")

# 提取前20个重要性得分最高的特征进行下一步处理
print(f"Selected top 20 features data shape: {selected_data_top20.shape}")

# 现在确保保存的是特征数据和标签
with pd.HDFStore('balanced_Y_NPQ_features_TOP20.h5', mode='w') as store:
    store.put('features', selected_data_top20, format='table')  # 保存特征数据
    store.put('labels', y2_series, format='table')  # 保存标签

print("Balanced and reduced features and labels saved")
