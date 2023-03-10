import pandas as pd
import glob

# # 获取所有CSV文件的文件名
# file_names = glob.glob("./train_data/*.csv")
# print(file_names)
# # 创建一个空的DataFrame对象
# merged_df = pd.DataFrame()

# # 循环读取每个CSV文件并将其合并到DataFrame中
# for file_name in file_names:
#     df = pd.read_csv(file_name)
#     merged_df = merged_df.append(df)

# # 将合并后的数据写入新的CSV文件中
# merged_df.to_csv("merged.csv", index=False)
df = pd.read_csv('data.csv', encoding='utf-8')
print(df['col3'])