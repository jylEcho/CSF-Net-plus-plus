import os
import h5py
import nibabel as nib
import numpy as np
import time
import shutil

# 定义路径和参数
test_data_path = 
test_data_file = 
coarse_liver_seg_path = 
test_label_path = 
output_path = 
window_min = -70
window_max = 180


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
# 窗口标准化函数
def apply_window_normalize(image, window_min, window_max):
    image = np.clip(image, window_min, window_max)
    return (image - window_min) / (window_max - window_min)

# 读取测试数据列表
def read_test_data_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

# 计算边界框
def calculate_bounding_boxes(mask, target_label=1):
    case_box = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]
    crop_boxes = {}
    for z in range(mask.shape[2]):
        positions = np.where(mask[:, :, z] == target_label)
        if positions[0].size > 0:
            min_x, max_x = np.min(positions[0]), np.max(positions[0])
            min_y, max_y = np.min(positions[1]), np.max(positions[1])
            crop_boxes[z + 1] = [min_x, max_x, min_y, max_y]
            case_box[0] = min(case_box[0], min_x)
            case_box[1] = max(case_box[1], max_x)
            case_box[2] = min(case_box[2], min_y)
            case_box[3] = max(case_box[3], max_y)
            case_box[4] = min(case_box[4], z)
            case_box[5] = max(case_box[5], z)
    return case_box, crop_boxes
# 处理测试文件
def process_test_files():
    start_time = time.time()
    clear_folder(output_path)
    test_files = read_test_data_list(test_data_file)

    for file_name in test_files:
        h5_file_data = {}  # 存储H5文件数据

        for phase in ['A', 'D', 'V']:  # 对每个期进行处理
            full_file_name = file_name + '-' + phase  # 构造完整的文件名
            data_img = nib.load(os.path.join(test_data_path, full_file_name + '.nii.gz'))
            image = apply_window_normalize(data_img.get_fdata(), window_min, window_max)
            h5_file_data[f'image_{phase}'] = image  # 存储每期的图像

            if phase == 'V':  # 只对一个期计算case_box和box_keys
                liver_seg_img = nib.load(os.path.join(coarse_liver_seg_path, full_file_name + '.nii.gz'))
                case_box, crop_boxes = calculate_bounding_boxes(liver_seg_img.get_fdata())
                h5_file_data['case_box'] = case_box
                keys = np.array(list(crop_boxes.keys()))
                values = np.array(list(crop_boxes.values()))
                h5_file_data['box_keys'] = keys
                h5_file_data['box_values'] = values

                label_img = nib.load(os.path.join(test_label_path, full_file_name + '.nii.gz'))
                label = label_img.get_fdata()
                h5_file_data[f'label'] = label  # 存储每期的标签

        # 保存H5文件
        with h5py.File(os.path.join(output_path, file_name + '.h5'), 'w') as h5_file:
            for key, data in h5_file_data.items():
                h5_file.create_dataset(key, data=data)

    end_time = time.time()
    print(f"处理完成。总耗时: {end_time - start_time:.2f}秒。")

process_test_files()
