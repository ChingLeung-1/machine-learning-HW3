from keras.engine.saving import load_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG

# model即为要可视化的网络模型
model = load_model("J:/PythonWorkSpace/HW3/keras/model2.h5")
SVG(model_to_dot(model).create(prog='dot', format='svg'))
