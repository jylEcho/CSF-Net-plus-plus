import os
import random

# 文件路径
data_path = '/path to your data fils (.nii.gz)'
output_path = '/output path'
# 设置测试集的病人数量
num_test_patients = XXX  # 你可以设置这个值为任意整数

# 获取所有文件名
files = os.listdir(data_path)

# 提取病人编号
patients = set()
for file in files:
    patient_id = '-'.join(file.split('-')[:2])  # 提取病名-XXX
    patients.add(patient_id)

# 检查是否有足够的病人进行分配
if num_test_patients > len(patients) // 2:
    raise ValueError("测试集的病人数量过多！")

# 将病人随机分为测试集和训练集
patients = list(patients)
random.shuffle(patients)
test_patients = patients[:num_test_patients]
train_patients = patients[num_test_patients:]

# 写入文件
with open(os.path.join(output_path, 'test_patients.txt'), 'w') as f:
    for patient in test_patients:
        f.write(patient + '\n')

with open(os.path.join(output_path, 'train_patients.txt'), 'w') as f:
    for patient in train_patients:
        f.write(patient + '\n')

print("完成!")
