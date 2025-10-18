''' 该py文件将data_path随机选择n个病人的v期转换成2d npz文件格式，每个npz有image_A/D/V 和 label两个key。
把剩下的病人转为h5格式放在output_h5_path '''

#data_path下的文件为”lesion-编号-A/D/V.nii.gz“,mask_path一样。

import os
import re
import random
import nibabel as nib
import numpy as np
# import h5py
import shutil
import time

# 定义路径和参数
data_path = ''
mask_path = ''
output_2d_path = ''
# output_h5_path = ''

train_patients_path = ''
liver_boxes_path = ''
window_min = -70 #-200
window_max = 180 #250

def apply_window_normalize(image, window_min, window_max):
    image = np.clip(image, window_min, window_max)
    return (image - window_min) / (window_max - window_min)

def read_train_patients(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def read_liver_boxes(file_path):
    liver_boxes = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            patient = parts[0]
            box = list(map(int, parts[1:]))
            liver_boxes[patient] = box
    return liver_boxes

def calculate_bounding_box(mask, target_label):
    positions = np.where(mask == target_label)
    if positions[0].size == 0:
        return None
    min_x, max_x = np.min(positions[0]), np.max(positions[0])
    min_y, max_y = np.min(positions[1]), np.max(positions[1])
    return [min_x, max_x, min_y, max_y]

def save_as_npz(images, mask, box, case_box, file_name):
    np.savez_compressed(os.path.join(output_2d_path, file_name),
                        image_A=images['A'],
                        image_D=images['D'],
                        image_V=images['V'],
                        label=mask,
                        box=box,
                        case_box=case_box)

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

def process_files():
    start_time = time.time()

    clear_folder(output_2d_path)

    train_patients = read_train_patients(train_patients_path)
    liver_boxes = read_liver_boxes(liver_boxes_path)

    for patient in train_patients:
        lesion, patient_number = patient.split('-')
        case_box = liver_boxes.get(patient, None) #min_x,max_x,min_y,max_y,min_z,max_z
        if case_box is None:
            continue  # 如果没有找到肝脏边界框，则跳过这个病人

        images = {}
        for phase in ['A', 'D', 'V']:
            file_name = f"{lesion}-{patient_number}-{phase}.nii.gz"
            data_img = nib.load(os.path.join(data_path, file_name))
            images[phase] = apply_window_normalize(data_img.get_fdata(), window_min, window_max)

        mask_file = os.path.join(mask_path, f"{lesion}-{patient_number}-V.nii.gz")
        mask_img = nib.load(mask_file)
        mask_slices = mask_img.get_fdata()

        for i in range(mask_slices.shape[2]):
            slice_mask = mask_slices[:, :, i]
            tumor_box = calculate_bounding_box(slice_mask, 2)
            liver_box = calculate_bounding_box(slice_mask, 1)
            if tumor_box is not None:
                box = tumor_box
            elif liver_box is not None and random.random() <= liver_slice_probability:
                box = liver_box
            else:
                continue

            save_as_npz({phase: images[phase][:, :, i] for phase in images}, slice_mask, box, case_box, f"{lesion}-{patient_number}-V-{i:04d}.npz")

    end_time = time.time()
    print(f"处理完成。总耗时: {end_time - start_time:.2f}秒。")

process_files()
