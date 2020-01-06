# 将label和像素数据分离
import pandas as pd

# 修改为train.csv在本地的相对或绝对地址
path = 'J:/PythonWorkSpace/HW3/OriginData/train.csv'
# 读取数据
df = pd.read_csv(path)
# 提取label数据
df_y = df[['label']]
# 提取feature（即像素）数据
df_x = df[['feature']]
# 将label写入label.csv
df_y.to_csv('J:/PythonWorkSpace/HW3/OriginData/label.csv', index=False, header=False)
# 将feature数据写入data.csv
df_x.to_csv('J:/PythonWorkSpace/HW3/OriginData/data.csv', index=False, header=False)