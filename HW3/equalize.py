import cv2
import numpy as np
import matplotlib.pyplot as plt

# 第一步：读入图片
img = cv2.imread('J:/PythonWorkSpace/HW3/photo/0.jpg')
img2 = cv2.imread('J:/PythonWorkSpace/HW3/photo/10.jpg')

# 读取单通道的灰色图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 高斯模糊
face_Gus = cv2.GaussianBlur(img, (3, 3), 0)
face_Gus2 = cv2.GaussianBlur(img2, (3, 3), 0)

# 自适应均衡化
clahe = cv2.createCLAHE(clipLimit=2.0,
                        tileGridSize=(8, 8))
face_clahe = clahe.apply(img)
face_clahe2 = clahe.apply(img2)


# 第三步：使用plt.hist绘制像素直方图
plt.subplot(121)
plt.hist(img.ravel(), 256)
plt.subplot(122)
plt.hist(face_clahe.ravel(), 256)
plt.show()

# 第四步：使用cv2.imshow()绘值均衡化的图像
cv2.namedWindow("picture", 1)
cv2.imshow("picture", np.hstack((img, face_clahe, img2, face_clahe2)))
cv2.waitKey(0)