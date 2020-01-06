import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2

# 选择设备
def get_defult_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# 验证数据正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = pred.cpu()
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc


class FaceDataset(data.Dataset):
    # 初始化
    def __init__(self, root):
        print('开始FaceDataset的__init__')
        super(FaceDataset, self).__init__()
        # root为train或val文件夹的地址
        self.root = root
        # 读取data-label对照表中的内容
        df_path = pd.read_csv(root + '/dataset.csv', header=None, usecols=[0])  # 读取第一列文件名
        df_label = pd.read_csv(root + '/dataset.csv', header=None, usecols=[1])  # 读取第二列label
        # 将其中内容放入numpy，方便后期索引
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    # 读取图片，item为引导号
    def __getitem__(self, item):
        img = cv2.imread('J:/PythonWorkSpace/HW3/photo/' + self.path[item])
        # 读取单通道的灰色图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # 自适应均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
        face_clahe = clahe.apply(img)
        # 像素标准化 变成符合模型的1*48*48维
        face_normalized = face_clahe.reshape(1, 48, 48) / 255.0
        # 训练模型需要变成tensor类型
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = self.label[item]
        # 返回预处理好的图片数据和label
        # print('FaceDataset的__getitem__得到{}'.format(self.path[item]), label)
        return face_tensor, label

    # 返回数据集长度
    def __len__(self):
        return self.path.shape[0]


# 卷积神经网络模型
class FaceCNN(nn.Module):
    # 初始化
    def __init__(self):
        super(FaceCNN, self).__init__()
        print('开始FaceCNN的__init__')
        # 第一次卷积，池化
        self.conv1 = nn.Sequential(
            # in_channels:输入数据通道数 out_channels：输出通道数（卷积核通道数）  kernel_size:卷积核大小3*3  stride:每次移动单位（步长）  padding：填充像素
            # 输入数据：1*48*48  输出数据：64*48*48  (48-3+1*2)/(1+1) = 48
            # 推导式:out_channels = (in_channels-kernel_size+2*padding)/ stride+1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            # normalize
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),  # activateFunction：ReLU
            # Pooling后：64*48/2*48/2
            nn.MaxPool2d(kernel_size=2, stride=2),  # 不一定要用MaxPooling也可以用AveragePooling
        ).cuda()

        # 第二次卷积，池化
        self.conv2 = nn.Sequential(
            # input:64*24*24  output:128*24*24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:128*12*12
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).cuda()

        # 第三次卷积，池化
        self.conv3 = nn.Sequential(
            # input:128*12*12  output:256*12*12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:256*6*6
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).cuda()

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层用ModelB，做两次dropout把结果flaten成7种表情
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.6),  # (回头可以调)
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            # nn.Linear(in_features=1024, out_features=256),
            # nn.RReLU(inplace=True),
            # nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=256, out_features=7),
            # 不需要加softmax 在做crossentropy的时候已经完成了
        ).cuda()

    # 模型feedForward
    def forward(self, x):
        x = x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # 加载数据，分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    print('开始trainModel')
    model_parm = "J:/PythonWorkSpace/HW3/model3.pk1"
    # model = FaceCNN().cuda()
    model = torch.load(model_parm).cuda()
    # loss function
    loss_function = nn.CrossEntropyLoss().cuda()
    # 优化器 这里是gradientDescent
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    # 开始训练
    for epoch in range(epochs):
        # 记录训练过程中的loss
        loss = 0.0
        model.train().cuda()  # train
        for images, labels in train_loader:
            labels = labels.cuda()
            images = images.cuda()
            # 梯度归零
            optimizer.zero_grad()
            # 向前传播
            output = model.forward(images).cuda()
            # 计算loss
            loss = loss_function(output, labels).cuda()
            # backPropagation
            loss.backward()
            # 更新参数
            optimizer.step()
        # 打印每轮的loss
        loss = loss.cpu()
        print('After {} epochs, the loss is:'.format(epoch + 1), loss.item())
        if epoch % 5 == 0:
            # 评估模型
            model.eval()
            # 训练正确率
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs, the acc_train is:'.format(epoch + 1), acc_train)
            print('After {} epochs, the acc_val is:'.format(epoch + 1), acc_val)

    return model


def main():
    print('开始main函数')
    # 训练数据
    train_dataset = FaceDataset(root='J:/PythonWorkSpace/HW3/train')
    # 验证数据集
    val_dataset = FaceDataset(root='J:/PythonWorkSpace/HW3/val')
    # 得到模型
    model = train(train_dataset, val_dataset, batch_size=1400, epochs=50, learning_rate=0.1, wt_decay=0)
    torch.save(model, 'J:/PythonWorkSpace/HW3/model4.pk1')


if __name__ == '__main__':
    main()
