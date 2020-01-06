import cv2
import os
import numpy as np
from keras import Model
from keras import backend as K
from keras.engine.saving import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


# 获取图片
def get_img(img_path, img_size):
    preimage = cv2.imread(img_path)
    image = cv2.resize(preimage, (48, 48), interpolation=cv2.INTER_LINEAR)
    preimage = cv2.resize(preimage, (200, 200), interpolation=cv2.INTER_LINEAR)
    # 读取单通道的灰色图
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # 自适应均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0,
                            tileGridSize=(8, 8))
    face_clahe = clahe.apply(image)
    # 像素标准化 变成符合模型的1*48*48维
    face_normalized = face_clahe.reshape(1, img_size, img_size, 1) / 255.0

    return preimage, image, face_normalized


def conv_output(model, i, img):
    layer_1 = K.function([model.layers[0].input], [model.layers[i].output])
    f1 = layer_1([img])[0]

    return f1


def model_predict(model):
    predict = model.predict(face)
    predict = np.argmax(predict, axis=1)
    text = ""
    if predict[0] == 0:
        text = "angry"
    elif predict[0] == 1:
        text = "disgust"
    elif predict[0] == 2:
        text = "terrify"
    elif predict[0] == 3:
        text = "happy"
    elif predict[0] == 4:
        text = "sad"
    elif predict[0] == 5:
        text = "shock"
    else:
        text = "normal"

    return text


def show_in_one(images, show_size=(720, 1080), blank_size=2, window_name="reslut"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("图片总数为： %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)
    cv2.imwrite('J:/PythonWorkSpace/HW3/testingResult.png', merge_img)


if __name__ == "__main__":

    model = load_model("J:/PythonWorkSpace/HW3/keras/model2.h5")
    path = 'J:/PythonWorkSpace/HW3/testingImages'
    dirs = os.listdir(path)
    imgs = []
    i = 0
    for file in dirs:
        preimg, image, face = get_img(img_path=path + "/" + file, img_size=48)
        text = model_predict(model)
        cv2.putText(preimg, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(preimg, file, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        imgs.append(preimg)
        # 获得某一层的输出
        intermediate_output = conv_output(model, 1, face)
        # print(model.get_layer('max_pooling2d_1').output)
        show_img = intermediate_output[:, :, :, 1]
        show_img.shape = [48, 48]
        plt.subplot(4, 4, i + 1)
        plt.imshow(show_img)
        plt.title(file, fontsize=8)
        plt.xticks([])
        plt.yticks([])
        i += 1

    plt.savefig("J:/PythonWorkSpace/HW3/layer1OutPut.png")
    plt.show()
    # 画出处理后的图像
    show_in_one(imgs)
    cv2.waitKey(0)
    # for i in range(len(model.layers)):
    #     print(model.get_layer(index=i).output)
