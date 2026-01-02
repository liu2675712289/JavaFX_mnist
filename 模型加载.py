import re
import torch
import numpy as np
from jinja2.optimizer import optimize

data=open('housing.data').readlines()
datas=[]
for item in data:
    out=re.sub(r"\s{2,}"," ",item).strip()
    # print(out)
    datas.append(out.split(" "))
datas=np.array(datas).astype(np.float32)
print(datas.shape)

Y=datas[:,-1]
X=datas[:,0:-1]

X_train=X[0:496,...]
Y_train=Y[0:496,...]
X_test=X[496:,...]
Y_test=Y[496:,...]

# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict=torch.nn.Linear(100,n_output)

    def forward(self,x):
        out=self.hidden(x)
        out=torch.relu(out)
        out=self.predict(out)
        return out

# PyTorch 2.6默认不支持加载自定义模型，需要设置成False
net=torch.load("model.pkl",weights_only=False)
#定义loss
loss_func=torch.nn.MSELoss()
X_data = torch.tensor(X_test, dtype=torch.float32)
Y_data = torch.tensor(Y_test, dtype=torch.float32)
pred = net.forward(X_data)
pred = torch.squeeze(pred)
loss_test = loss_func(pred, Y_data) * 0.001
print("loss_test:{}".format( loss_test))