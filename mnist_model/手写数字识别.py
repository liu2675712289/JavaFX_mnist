import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import  torch.utils.data as data_utils
from CNN import  CNN

#解析数据
train_data=dataset.MNIST(root="mnist",
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

test_data=dataset.MNIST(root="mnist",train=False,transform=transforms.ToTensor(),download=False)
#batchsize对数据分批读取
train_loader=data_utils.DataLoader(dataset=train_data,batch_size=64,shuffle=True)#shuffle打乱数据

test_loader=data_utils.DataLoader(dataset=test_data,batch_size=64,shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#搭建网络模型
cnn=CNN()
cnn=cnn.cuda()
#定义loss损失函数
loss_func=torch.nn.CrossEntropyLoss()
#定义优化器
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.01)
#训练模型获取优化参数
for epoch in range(300):
    for i, (images,labels) in enumerate(train_loader):
        images=images.cuda()
        labels=labels.cuda()

        outputs=cnn(images)
        loss=loss_func(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print("epoch is {},ite is{}/{},loss is {}".format(epoch+1,i,
    #                                                       len(train_data)//64,
    #                                                       loss.item()))
#样本测试
    loss_test=0
    accuracy=0
    for i, (images,labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        #[batchsize]
        #outputs=batchsize*cls_num
        loss_test += loss_func(outputs, labels)
        _,pred=outputs.max(1)
        accuracy+=(pred==labels).sum().item()

    accuracy=accuracy/len(test_data)
    loss_test=loss_test/(len(test_data)//64)

    print("epoch is {},accuracy is {},"
          "loss test is {}".format(epoch+1,accuracy,loss_test.item()))
#保存模型
x=torch.randn(1, 1, 28, 28).to(device)
torch.onnx.export(
    model=cnn,
    args=x,
    f="mnist_model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
)
# torch.save(cnn,"mnist_model.pkl")