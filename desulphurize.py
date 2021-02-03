# 导入库
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso
from torch.utils.data import Dataset, DataLoader
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# 忽略警告并设置GPU
warnings.filterwarnings('ignore')
gpu = torch.device('cuda:0')


# 变量重要性展示
def feature_importance(weight, title):
    plt.figure()
    plt.bar(range(X.shape[1]), weight)
    plt.xlabel('Process Variable')
    plt.ylabel('Importance')
    plt.title(title)


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, label, mode):
        self.data = data
        self.label = label
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, n_input, n_lstm, n_output, dropout=0.0):
        super(LSTM, self).__init__()
        self.n_input = n_input
        self.n_lstm = [n_input] + list(n_lstm) + [n_output]
        self.n_output = n_output
        self.dropout = dropout
        self.lstm = nn.ModuleList()

        # LSTM层
        for i in range(len(n_lstm) + 1):
            self.lstm.append(nn.LSTM(self.n_lstm[i], self.n_lstm[i + 1], dropout=dropout))

    def forward(self, x):
        feat = x

        for i in self.lstm:
            feat = i(feat)
            feat = feat[0]

        return feat


# 导入数据
data = pd.read_excel('乐清电厂烟气脱硫运行数据.xls', header=2, index_col=0).iloc[1:, :].astype(float)
X = data.drop(columns='#1净烟气SO2基准浓度')
y = np.array(data.loc[:, '#1净烟气SO2基准浓度']).reshape(-1, 1)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 数据标准化
scaler_X = StandardScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)
y_train_std = scaler_y.transform(y_train)

# 变量选择
rank = []
selector1 = SelectKBest(f_regression, k=5).fit(X, y)
selector2 = SelectKBest(mutual_info_regression, k=5).fit(X, y)
rank.append(selector1.scores_.argsort().argsort())
rank.append(selector2.scores_.argsort().argsort())
feature_importance(selector1.scores_, 'f_regression')
feature_importance(selector2.scores_, 'mutual_info_regression')

# Lasso
print('=====Lasso=====')
lasso = Lasso()
params_lasso = {'alpha': np.logspace(-3, 3, 7)}
reg = GridSearchCV(lasso, params_lasso, 'r2', iid=True, cv=5).fit(X_train, y_train)
y_train_lasso = reg.predict(X_train).reshape(-1, 1)
y_test_lasso = reg.predict(X_test).reshape(-1, 1)
r2_lasso = r2_score(y_test, y_test_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_test_lasso))
print('R2:{},RMSE:{}\n'.format(r2_lasso, rmse_lasso))
rank.append(np.abs(reg.best_estimator_.coef_).argsort().argsort())
feature_importance(np.abs(reg.best_estimator_.coef_), 'Lasso')

# SVR
print('=====SVR=====')
svr = LinearSVR(epsilon=0.0001)
params_svr = {'C': np.logspace(-3, 3, 7)}
reg = GridSearchCV(svr, params_svr, 'r2', iid=True, cv=5).fit(X_train, y_train)
y_train_svr = reg.predict(X_train).reshape(-1, 1)
y_test_svr = reg.predict(X_test).reshape(-1, 1)
r2_svr = r2_score(y_test, y_test_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_test_svr))
print('R2:{},RMSE:{}\n'.format(r2_svr, rmse_svr))
rank.append(np.abs(reg.best_estimator_.coef_).argsort().argsort())
feature_importance(np.abs(reg.best_estimator_.coef_), 'SVR')

# RandomForest
print('=====RandomForest=====')
rf = RandomForestRegressor()
params_rf = {'n_estimators': range(10, 210, 10), 'max_depth': range(3, 11)}
reg = RandomizedSearchCV(rf, params_rf, 50, 'r2', iid=True, cv=5).fit(X_train, y_train)
y_train_rf = reg.predict(X_train).reshape(-1, 1)
y_test_rf = reg.predict(X_test).reshape(-1, 1)
r2_rf = r2_score(y_test, y_test_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_rf))
print('R2:{},RMSE:{}\n'.format(r2_rf, rmse_rf))
rank.append(reg.best_estimator_.feature_importances_.argsort().argsort())
feature_importance(reg.best_estimator_.feature_importances_, 'RandomForest')

# XgBoost
print('=====XgBoost=====')
xgb = XGBRegressor(objective='reg:linear')
params_xgb = {'n_estimators': range(10, 210, 10), 'max_depth': range(3, 11),
              'learning_rate': np.logspace(-2, -1), 'colsample_bytree': np.linspace(0.5, 1)}
reg = RandomizedSearchCV(xgb, params_xgb, 50, 'r2', iid=True, cv=5).fit(X_train, y_train)
y_train_xgb = reg.predict(X_train).reshape(-1, 1)
y_test_xgb = reg.predict(X_test).reshape(-1, 1)
r2_xgb = r2_score(y_test, y_test_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_test_xgb))
print('R2:{},RMSE:{}\n'.format(r2_xgb, rmse_xgb))
rank.append(reg.best_estimator_.feature_importances_.argsort().argsort())
feature_importance(reg.best_estimator_.feature_importances_, 'XgBoost')
rank_avg = X.shape[1] - np.stack(rank).mean(axis=0)
print(rank_avg)
plt.figure()
plt.bar(range(X.shape[1]), rank_avg)
plt.xlabel('Process Variable')
plt.ylabel('Rank')
plt.title('Influential Rank of Process Variables')
plt.show()

# Fully Connected Networks
print('=====Fully Connected Networks=====')
reg = MLPRegressor(hidden_layer_sizes=(1024, 256)).fit(X_train, y_train_std)
y_train_fc = scaler_y.inverse_transform(reg.predict(X_train).reshape(-1, 1))
y_test_fc = scaler_y.inverse_transform(reg.predict(X_test).reshape(-1, 1))
r2_fc = r2_score(y_test, y_test_fc)
rmse_fc = np.sqrt(mean_squared_error(y_test, y_test_fc))
print('R2:{},RMSE:{}\n'.format(r2_fc, rmse_fc))

# Long Short-Term Memory
print('=====Long Short-Term Memory=====')
X_train_gpu = torch.tensor(X_train, dtype=torch.float32, device=gpu)
X_test_gpu = torch.tensor(X_test, dtype=torch.float32, device=gpu)
y_train_std_gpu = torch.tensor(y_train_std, dtype=torch.float32, device=gpu)

# 数据转化
seq_len = 20
n_epoch = 200
X_train_3d = []
y_train_3d = []
for i in range(X_train_gpu.shape[0] - seq_len + 1):
    X_train_3d.append(X_train_gpu[i:i + seq_len, :])
    y_train_3d.append(y_train_std_gpu[i:i + seq_len, :])
X_train_3d = torch.stack(X_train_3d, 1)
y_train_3d = torch.stack(y_train_3d, 1)

# 模型构建
n_lstm = (1024, 256)
lstm = LSTM(X_train.shape[1], n_lstm, 1).to(gpu)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(lstm.parameters(), lr=0.001, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

# 模型训练
lstm.train()
for epoch in range(n_epoch):
    loss_train = 0

    # 生成数据集和数据加载器
    data_train = MyDataset(X_train_3d, y_train_3d, '3D')
    data_loader = DataLoader(data_train, batch_size=64, shuffle=True)

    # 每个batch
    for (batch_X, batch_y) in data_loader:
        batch_X = batch_X.permute(1, 0, 2)
        batch_y = batch_y.permute(1, 0, 2)
        optimizer.zero_grad()
        output = lstm(batch_X)
        loss = criterion(output, batch_y)
        loss_train += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()

    # 打印结果
    print('Epoch:{:3d} Loss:{:4f}'.format(epoch + 1, loss_train))
print('Optimization finished!')

# 测试结果
lstm.eval()
y_train_lstm = scaler_y.inverse_transform(
    lstm(X_train_gpu.view(-1, 1, X_test.shape[1])).cpu().detach().numpy().reshape(-1, 1))
y_test_lstm = scaler_y.inverse_transform(
    lstm(X_test_gpu.view(-1, 1, X_test.shape[1])).cpu().detach().numpy().reshape(-1, 1))
r2_lstm = r2_score(y_test, y_test_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_test_lstm))
print('R2:{},RMSE:{}\n'.format(r2_lstm, rmse_lstm))

# 预测情况图
plt.figure()
plt.plot(y_train)
plt.plot(y_train_lasso)
plt.plot(y_train_svr)
plt.plot(y_train_rf)
plt.plot(y_train_xgb)
plt.plot(y_train_fc)
plt.plot(y_train_lstm)
plt.legend(
    ['Ground Truth', 'Lasso', 'SVR', 'RandomForest', 'XgBoost', 'Fully Connected Networks', 'Long Short-Term Memory'])
plt.xlabel('Sample')
plt.ylabel('SO2 Concentration')
plt.title('Train Set')
plt.figure()
plt.plot(y_test)
plt.plot(y_test_lasso)
plt.plot(y_test_svr)
plt.plot(y_test_rf)
plt.plot(y_test_xgb)
plt.plot(y_test_fc)
plt.plot(y_test_lstm)
plt.legend(
    ['Ground Truth', 'Lasso', 'SVR', 'RandomForest', 'XgBoost', 'Fully Connected Networks', 'Long Short-Term Memory'])
plt.xlabel('Sample')
plt.ylabel('SO2 Concentration')
plt.title('Test Set')
plt.show()
