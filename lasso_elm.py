# 导入库
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index, :], self.label[index, :]

    def __len__(self):
        return self.data.shape[0]


# 自定义Loss函数
class MyLoss(nn.Module):
    def __init__(self, mu=0.1):
        super(MyLoss, self).__init__()
        self.mu = mu

    def forward(self, y_true, y_pred, model):
        term1 = torch.norm(y_true - y_pred, 2) ** 2
        term2 = 0
        term3 = 0
        for i in range(model.w1.weight.shape[1]):
            term2 += torch.norm(model.w1.weight[:, i], 2)
        for i in range(model.w2.weight.shape[0]):
            term3 += torch.norm(model.w2.weight[:, i], 2)
        return term1 + self.mu * (term2 + term3)


# 极限学习机
class RVFLNN(nn.Module):
    def __init__(self, n_input, n_output, n_hid=128, direct_link=True):
        super(RVFLNN, self).__init__()
        self.n_input = n_input
        self.n_hid = n_hid
        self.n_output = n_output
        self.dl = direct_link
        self.w1 = nn.Linear(n_input, n_hid)
        if self.dl:
            self.w2 = nn.Linear(n_input + n_hid, n_output)
        else:
            self.w2 = nn.Linear(n_hid, n_output)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.w1(x))
        if self.dl:
            y = self.w2(torch.cat((h, x), 1))
        else:
            y = self.w2(h)
        return y


# RVFLNN模型
class RvflnnModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_input, n_output, n_hid=128, direct_link=True, n_epoch=200, lr=0.001, weight_decay=0.1,
                 step_size=50, gamma=0.8, mu=0.1):
        super(RvflnnModel, self).__init__()
        self.n_epoch = n_epoch
        self.gpu = torch.device('cuda:1')
        self.model = RVFLNN(n_input, n_output, n_hid, direct_link).to(self.gpu)
        self.criterion = MyLoss(mu=mu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.loss_hist = []

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=self.gpu)
        y = torch.tensor(y, dtype=torch.float32, device=self.gpu)

        # 模型训练
        self.model.train()
        for epoch in range(self.n_epoch):
            loss_train = 0

            # 生成数据集和数据加载器
            data_train = MyDataset(X, y)
            data_loader = DataLoader(data_train, batch_size=64, shuffle=True)

            # 每个batch
            for (batch_X, batch_y) in data_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(batch_y, output, self.model)
                loss_train += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            self.loss_hist.append(loss_train)

            # 打印结果
            print('Epoch:{:3d} Loss:{:4f}'.format(epoch + 1, loss_train))
        print('Optimization finished!')

        return self

    def predict(self, X):
        self.model.eval()
        y = self.model(torch.tensor(X, dtype=torch.float32, device=self.gpu))
        return y.cpu().detach().numpy()


# 导入数据
data = pd.read_excel('乐清电厂烟气脱硫运行数据.xls', header=2, index_col=0).iloc[1:, :].astype(float)
X = data.drop(columns='#1净烟气SO2基准浓度')
y = np.array(data.loc[:, '#1净烟气SO2基准浓度']).reshape(-1, 1)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 数据标准化
scaler_X = MinMaxScaler().fit(X_train)
scaler_y = MinMaxScaler().fit(y_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)
y_train_std = scaler_y.transform(y_train)

# RVFLNN
print('=====RVFLNN=====')
reg = RvflnnModel(X.shape[1], y.shape[1], weight_decay=0.2, mu=0.1).fit(X_train, y_train_std)
y_fit = scaler_y.inverse_transform(reg.predict(X_train))
y_pred = scaler_y.inverse_transform(reg.predict(X_test))
r2_fit = r2_score(y_train, y_fit)
r2_pred = r2_score(y_test, y_pred)
rmse_fit = np.sqrt(mean_squared_error(y_train, y_fit))
rmse_pred = np.sqrt(mean_squared_error(y_test, y_pred))

# 打印结果
print('Train:', r2_fit, rmse_fit)
print('Test:', r2_pred, rmse_pred)
sns.heatmap(np.abs(reg.model.w1.weight.cpu().detach().numpy()))
plt.figure()
sns.heatmap(np.abs(reg.model.w2.weight.cpu().detach().numpy()))
plt.figure()
plt.plot(y_fit)
plt.plot(y_train)
plt.figure()
plt.plot(y_pred)
plt.plot(y_test)
plt.show()
