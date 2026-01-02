import re
import torch
import numpy as np

'''
数据>网络结构>损失>优化>测试>推理
'''

#解析数据data
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
#搭建网络Net(回归网络)
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

net=Net(13,1)
#定义loss
loss_func=torch.nn.MSELoss()
#定义优化器optimiter
optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
#训练模型获取参数training
for i in range(10000):
    X_data=torch.tensor(X_train,dtype=torch.float32)
    Y_data=torch.tensor(Y_train,dtype=torch.float32)
    pred=net.forward(X_data)
    pred=torch.squeeze(pred)
    loss=loss_func(pred,Y_data)*0.001

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("ite:{},loss_train:{}".format(i,loss))
    print(pred[0:10])
    print(Y_data[0:10])
#样本测试
    X_data = torch.tensor(X_test, dtype=torch.float32)
    Y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(X_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, Y_data) * 0.001
    print("ite:{},loss_test:{}".format(i, loss_test))


# torch.save(net, "model.pkl")#保存模型
# torch.save(net.state_dict(),"params.pkl")#保存参数
