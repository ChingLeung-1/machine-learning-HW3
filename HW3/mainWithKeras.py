import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator, load_img
# 搭积木的桌子
from keras.models import Sequential, load_model
# 各种形状的积木
from keras.layers import Conv2D, MaxPooling2D
# 输出层积木
from keras.layers import Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adadelta, adam
from keras.layers.advanced_activations import LeakyReLU


# 获取图片
def get_img(img_path, img_size):
    img = cv2.imread('J:/PythonWorkSpace/HW3/photo/' + img_path)
    # 读取单通道的灰色图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 自适应均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0,
                            tileGridSize=(8, 8))
    face_clahe = clahe.apply(img)
    # 像素标准化 变成符合模型的1*48*48维
    face_normalized = face_clahe.reshape(img_size, img_size, 1) / 255.0

    return face_normalized


# 根据batch大小返回数据
def data_generator(datapath, bs, lb, img_size=48, mode="train", aug=None):
    file = open(datapath + '/dataset.csv', 'r', )
    while 1:
        # 初始化参数
        images = []
        labels = []
        while len(images) < bs:
            # 尝试读下一行
            line = file.readline()
            # 如果已经读到最后一行
            if line == "":
                # 重置文档到开头
                file.seek(0)
                # 重新开始读取
                line = file.readline()
                # 如果batch已经满了 就需要停止

                # 重新开始获取数据
                if mode == "eval":
                    break
            # 提取label
            line = line.strip().split(",")
            label = line[1]
            image_path = line[0]
            image = get_img(image_path, img_size)

            # 添加到当前batch中
            labels.append(label)
            images.append(image)
        # 准备好一个batch之后
        labels = lb.transform(np.array(labels))

        if aug is not None:
            (images, labels) = next(aug.flow(np.array(images), labels, batch_size=bs))

        # 把结果回传
        yield np.array(images), labels


def faceCNN():
    model = Sequential()
    # 第一次卷积  outputChannel=64  filter=3*3  激活函数:linear 输入：48*48*1
    model.add(Conv2D(64, (3, 3), strides=1, padding="same", input_shape=[48, 48, 1]))
    model.add(LeakyReLU(alpha=0.1))

    # 第一次池化 池化后64 * 24 * 24
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # 第二次卷积 outputChannel=128  filter=3*3  激活函数:linear 输入：24*24*64
    model.add(Conv2D(128, (3, 3), strides=1, padding="same", input_shape=[24, 24, 64]))
    model.add(LeakyReLU(alpha=0.1))

    # 第二次池化 池化后128 * 12 * 12
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # 第三次卷积
    model.add(Conv2D(256, (3, 3), strides=1, padding="same", input_shape=[12, 12, 128]))
    model.add(LeakyReLU(alpha=0.1))

    # 第三次池化
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(7, activation="softmax"))

    model.compile(loss=categorical_crossentropy,
                  # optimizer=SGD(lr=1e-2, momentum=0.9, decay=1e-2 / epoch),
                  optimizer=adam(),
                  metrics=['accuracy'])
    return model


def train(model, batch_size, epoch):
    # 记录训练和测试的所有图片
    NUM_TRAIN_IMAGES = 0
    labels = set()
    NUM_TEST_IMAGES = 0
    testLabels = []

    # 读取训练集中的所有标签方便做二进制化
    f = open('J:/PythonWorkSpace/HW3/train/dataset.csv', "r")
    # 循环所有行
    for line in f:
        # 提取出标签的分类
        # 记录所有训练图象集
        label = line.strip().split(",")[1]
        labels.add(label)
        NUM_TRAIN_IMAGES += 1
    f.close()

    # 开始统计测试集
    # 读取训练集中的所有标签方便做二进制化
    f = open('J:/PythonWorkSpace/HW3/val/dataset.csv', "r")
    # 循环所有行
    for line in f:
        # 提取出标签的分类
        # 记录所有训练图象集
        label = line.strip().split(",")[1]
        testLabels.append(label)
        NUM_TEST_IMAGES += 1
    f.close()

    # 标签二进制化
    lb = LabelBinarizer()
    lb.fit(list(labels))
    testLabels = lb.transform(testLabels)

    # 图像强化 意味着图像不是静止的
    aug = ImageDataGenerator(rotation_range=15, zoom_range=0.1,
                             width_shift_range=0.15, height_shift_range=0.15, shear_range=0.15,
                             horizontal_flip=True, fill_mode="nearest")

    # 初始化训练数据  只有train会做数据增强
    trainGen = data_generator('J:/PythonWorkSpace/HW3/train/', batch_size, lb, mode="train", aug=aug)
    testGen = data_generator('J:/PythonWorkSpace/HW3/val/', batch_size, lb, mode="train", aug=None)

    # 开始训练
    print("[INFO] training w/ generator...")
    H = model.fit_generator(trainGen,
                            steps_per_epoch=NUM_TRAIN_IMAGES // batch_size,
                            validation_data=testGen,
                            validation_steps=NUM_TEST_IMAGES // batch_size,
                            epochs=epoch)

    # 评估结果
    testGen = data_generator('J:/PythonWorkSpace/HW3/val/', batch_size, lb,
                             mode="eval", aug=None)
    # 根据模型找到当前最有可能的标签
    predIdxs = model.predict_generator(testGen,
                                       steps=(NUM_TEST_IMAGES // batch_size) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)

    # 输出报告
    print("[INFO] evaluating network...")
    print(classification_report(testLabels.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))
    # 最后保存model
    model.save('J:/PythonWorkSpace/HW3/keras/model2.h5')
    model.save_weights('J:/PythonWorkSpace/HW3/keras/modelWight2.h5')
    # 最后画出训练字典H 和 matplotlib
    N = epoch
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("J:/PythonWorkSpace/HW3/keras/plot2.png")


if __name__ == "__main__":
    # 初始化参数
    batch_size = 1500
    epoch = 100
    model = faceCNN()
    model.load_weights('J:/PythonWorkSpace/HW3/keras/modelWight.h5')
    train(model, batch_size, epoch)
