import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, cohen_kappa_score, confusion_matrix

# 加载 HDF5 文件中的分类和回归任务特征与标签
with pd.HDFStore('classification_Y_II_features_TOP20.h5', mode='r') as store:
    X_classification = store['x_c'].values
    y_classification = store['y_c'].values

with pd.HDFStore('regression_Y_II_features_TOP20.h5', mode='r') as store:
    X_regression = store['x_r'].values
    y_regression = store['y_r'].values

# 数据标准化
scaler_class = StandardScaler()
X_classification = scaler_class.fit_transform(X_classification)

scaler_reg = StandardScaler()
X_regression = scaler_reg.fit_transform(X_regression)

# 划分数据集（80% 训练，20% 测试）
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42)

# 创建多任务学习模型（分类和回归特征不一致，添加共享层）
def create_multitask_model(input_dim_class, input_dim_reg):
    # 输入层：针对分类特征和回归特征分别定义输入
    input_class = Input(shape=(input_dim_class, 1), name='input_class')
    input_reg = Input(shape=(input_dim_reg, 1), name='input_reg')

    # 分类任务特有的特征处理
    x_class = Conv1D(filters=32, kernel_size=3, activation='relu')(input_class)
    x_class = BatchNormalization()(x_class)
    x_class = Dropout(0.5)(x_class)
    x_class = Flatten()(x_class)
    x_class = Dense(128, activation='relu')(x_class)
    x_class = BatchNormalization()(x_class)
    x_class = Dropout(0.5)(x_class)

    # 回归任务特有的特征处理
    x_reg = Conv1D(filters=32, kernel_size=3, activation='relu')(input_reg)
    x_reg = BatchNormalization()(x_reg)
    x_reg = Dropout(0.5)(x_reg)
    x_reg = Flatten()(x_reg)
    x_reg = Dense(128, activation='relu')(x_reg)
    x_reg = BatchNormalization()(x_reg)
    x_reg = Dropout(0.5)(x_reg)

    # 共享层：将分类和回归的特征组合后提取共享特征
    combined_features = Concatenate()([x_class, x_reg])
    shared_layer = Dense(128, activation='relu')(combined_features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Dropout(0.5)(shared_layer)

    # 分类分支：进一步处理共享层特征
    classification_branch = Dense(64, activation='relu')(shared_layer)
    classification_branch = BatchNormalization()(classification_branch)
    classification_branch = Dropout(0.5)(classification_branch)
    classification_output = Dense(1, activation='sigmoid', name='classification_output')(classification_branch)

    # 回归分支：进一步处理共享层特征
    regression_branch = Dense(64, activation='relu')(shared_layer)
    regression_branch = BatchNormalization()(regression_branch)
    regression_branch = Dropout(0.5)(regression_branch)
    regression_output = Dense(1, activation='linear', name='regression_output')(regression_branch)

    # 精准判别分支：使用分类和回归结果提升对0.48附近的精度
    combined_input = Concatenate()([classification_output, regression_output])
    precision_branch = Dense(16, activation='relu')(combined_input)
    precision_output = Dense(1, activation='sigmoid', name='precision_output')(precision_branch)

    # 创建多任务模型，包含分类、回归和精准判别三个输出
    model = Model(inputs=[input_class, input_reg], outputs=[classification_output, regression_output, precision_output])
    return model

# 获取输入维度
input_dim_class = X_class_train.shape[1]
input_dim_reg = X_reg_train.shape[1]

# 调整输入数据的形状以适应 Conv1D 层
X_class_train = np.expand_dims(X_class_train, axis=2)
X_class_test = np.expand_dims(X_class_test, axis=2)
X_reg_train = np.expand_dims(X_reg_train, axis=2)
X_reg_test = np.expand_dims(X_reg_test, axis=2)

model = create_multitask_model(input_dim_class, input_dim_reg)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss={'classification_output': 'binary_crossentropy',  # 分类损失函数
                    'regression_output': 'mean_squared_error',  # 回归损失函数
                    'precision_output': 'binary_crossentropy'},  # 精准判别损失函数
              metrics={'classification_output': 'accuracy',  # 分类评估指标
                       'regression_output': 'mean_absolute_error',  # 回归评估指标
                       'precision_output': 'accuracy'})  # 精准判别的准确率

# 定义早停机制，防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(
    {'input_class': X_class_train, 'input_reg': X_reg_train},
    {'classification_output': y_class_train, 'regression_output': y_reg_train, 'precision_output': y_class_train},
    validation_data=({'input_class': X_class_test, 'input_reg': X_reg_test},
                     {'classification_output': y_class_test, 'regression_output': y_reg_test, 'precision_output': y_class_test}),
    epochs=20, batch_size=32, callbacks=[early_stopping])

# 评估模型在测试集上的表现
loss, classification_loss, regression_loss, precision_loss, classification_acc, regression_mae, precision_acc = model.evaluate(
    {'input_class': X_class_test, 'input_reg': X_reg_test},
    {'classification_output': y_class_test, 'regression_output': y_reg_test, 'precision_output': y_class_test})

print(f"Test Classification Accuracy: {classification_acc}")
print(f"Test Regression Mean Absolute Error: {regression_mae}")
print(f"Test Precision Accuracy: {precision_acc}")

# 预测分类、回归和精准判别任务
classification_pred, regression_pred, precision_pred = model.predict({'input_class': X_class_test, 'input_reg': X_reg_test})

# 将分类概率转换为二进制标签（大于0.5的为1）
classification_pred_binary = (classification_pred > 0.5).astype(int).flatten()
precision_pred_binary = (precision_pred > 0.5).astype(int).flatten()

# 计算分类任务的 Kappa 系数
classification_kappa = cohen_kappa_score(y_class_test, classification_pred_binary)
precision_kappa = cohen_kappa_score(y_class_test, precision_pred_binary)

# 计算分类任务的混淆矩阵
classification_conf_matrix = confusion_matrix(y_class_test, classification_pred_binary)
precision_conf_matrix = confusion_matrix(y_class_test, precision_pred_binary)

# 计算回归任务的 R² 和 RMSE
r2 = r2_score(y_reg_test, regression_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, regression_pred))

print(f"Test Classification Kappa: {classification_kappa}")
print(f"Test Precision Kappa: {precision_kappa}")
print(f"Test Regression R²: {r2}")
print(f"Test Regression RMSE: {rmse}")

# 评估模型在训练集上的表现
classification_pred_train, regression_pred_train, precision_pred_train = model.predict({'input_class': X_class_train, 'input_reg': X_reg_train})

# 将训练集的分类概率转换为二进制标签（大于0.5的为1）
classification_pred_train_binary = (classification_pred_train > 0.5).astype(int).flatten()
precision_pred_train_binary = (precision_pred_train > 0.5).astype(int).flatten()

# 计算训练集的分类精度和 Kappa 系数
classification_train_acc = accuracy_score(y_class_train, classification_pred_train_binary)
classification_train_kappa = cohen_kappa_score(y_class_train, classification_pred_train_binary)

precision_train_acc = accuracy_score(y_class_train, precision_pred_train_binary)
precision_train_kappa = cohen_kappa_score(y_class_train, precision_pred_train_binary)

# 计算训练集回归任务的 R² 和 RMSE
r2_train = r2_score(y_reg_train, regression_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_reg_train, regression_pred_train))

# 输出训练集和测试集的结果
print("\nTraining Set Results:")
print(f"Train Classification Accuracy: {classification_train_acc}")
print(f"Train Classification Kappa: {classification_train_kappa}")
print(f"Train Regression R²: {r2_train}")
print(f"Train Regression RMSE: {rmse_train}")
print(f"Train Precision Accuracy: {precision_train_acc}")
print(f"Train Precision Kappa: {precision_train_kappa}")

print("\nTest Set Results:")
print(f"Test Classification Accuracy: {classification_acc}")
print(f"Test Classification Kappa: {classification_kappa}")
print(f"Test Regression R²: {r2}")
print(f"Test Regression RMSE: {rmse}")
print(f"Test Precision Accuracy: {precision_acc}")
print(f"Test Precision Kappa: {precision_kappa}")

# 保存模型
model.save('Y_II_3_model.h5')
print("Multitask model with shared layer saved to 'Y_II_2_model.h5'")

# 保存测试集的实际值和预测值
test_results = pd.DataFrame({
    'Actual': y_reg_test.flatten(),
    'Predicted': regression_pred.flatten()
})
test_results.to_csv('test_results_Y_II_3.csv', index=False)
print("Testing results saved to 'test_results_Y_II_2.csv'")