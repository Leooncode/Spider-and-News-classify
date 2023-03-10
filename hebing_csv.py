import csv
import pandas as pd

# 读取CSV文件
with open('data.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = [row for row in reader]

# 合并两列数据
merged_data = [col[0] + ' ' + col[1] for col in data]

# 将结果写入CSV文件
df = pd.DataFrame({'Merged Column': merged_data})
df.to_csv('output_file.csv', index=False)
