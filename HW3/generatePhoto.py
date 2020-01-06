import cv2
import numpy as np

# 读取像素数据
data = np.loadtxt('J:/PythonWorkSpace/HW3/OriginData/data.csv')

# 按行取数据
for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48)) # reshape
    cv2.imwrite('J:/PythonWorkSpace/HW3/photo/' + '{}.jpg'.format(i), face_array) # 写图片