import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import  center_of_mass
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa


def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug, img, seg ):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic() 
    image_aug = aug_det.augment_image( img )

    segmap = ia.SegmentationMapOnImage( seg , nb_classes=np.max(seg)+1 , shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug , segmap_aug

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Multiphase_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, norm_x_transform=None, norm_y_transform=None,crop_train='max_crop'):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split

        self.crop_train = crop_train
        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def random_crop(self, slice, box, case_box, output_size):
        """在给定的肝脏框内进行随机裁剪，同时考虑整个病例的范围"""
        min_x, max_x, min_y, max_y = box
        min_x_case, max_x_case, min_y_case, max_y_case, _, _ = case_box


        # 确保裁剪中心点的范围是有效的
        if x_center_max < x_center_min or y_center_max < y_center_min:
            # 如果范围无效，则返回原始切片和空坐标
            return  (0, 0, slice.shape[1], slice.shape[0])

        # 随机选择裁剪中心点
        x_center = np.random.randint(x_center_min, x_center_max)
        y_center = np.random.randint(y_center_min, y_center_max)

        # 计算裁剪的起始和结束点
        x_start = max(x_center - output_size // 2, 0)
        y_start = max(y_center - output_size // 2, 0)
        x_end = min(x_start + output_size, slice.shape[0])
        y_end = min(y_start + output_size, slice.shape[1])

        return  (x_start, y_start, x_end, y_end)

    def random_crop_2(self, slice, box, case_box, output_size):
        """在给定的肝脏框内进行随机裁剪，同时考虑整个病例的范围"""
        # 在当前切片的感兴趣区域内随机选择中心点
        min_x, max_x, min_y, max_y = box
        # if min_x >= max_x or min_y >= max_y:
        #     print("错误！！！！！")

        x_center = np.random.randint(min_x, max_x + 1)  # +1避免出现一个像素的情况
        y_center = np.random.randint(min_y, max_y + 1)  # +1避免出现一个像素的情况

        # 随机缩放和裁剪
        scale = np.random.uniform(0.8, 1.2)
        crop_size = int(self.img_size * scale)

        # 计算裁剪的起始和结束点，确保它们既在切片内，又在整个病例的肝脏区域内
        x_start = min(max(case_box[0] + crop_size // 2, x_center), case_box[1] - crop_size // 2 - 1)
        y_start = min(max(case_box[2] + crop_size // 2, y_center), case_box[3] - crop_size // 2 - 1)

        # 考虑到裁剪的区域可能越出图像边界，需要进行调整
        x_start = max(x_start - crop_size // 2, 0)
        y_start = max(y_start - crop_size // 2, 0)
        x_end = min(x_start + crop_size, slice.shape[0])
        y_end = min(y_start + crop_size, slice.shape[1])
        return  (x_start, y_start, x_end, y_end)
    def max_crop(self, slice, case_box, output_size):
        """在给定的肝脏框内进行最大可用区域裁剪"""
        min_x_case, max_x_case, min_y_case, max_y_case, _, _ = case_box

        # 考虑输出尺寸，避免裁剪区域超出图像边界
        x_start = max(min_x_case, 0)
        y_start = max(min_y_case, 0)
        # x_end = min(max_x_case, slice.shape[1] - output_size[0])
        x_end = max_x_case
        y_end = max_y_case


        # 执行裁剪
        # cropped_slice = slice[x_start:x_end, y_start:y_end]
        return  (x_start, y_start, x_end, y_end)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image_V, image_A, image_D, label = data['image_V'].astype(np.float32), data['image_A'].astype(np.float32), \
                                               data['image_D'].astype(np.float32), data['label'].astype(np.float32)

            # 加载整个病例的肝脏区域框和当前切片的感兴趣区域框
            case_box = data['case_box']  # 整个病例的肝脏区域框，假设为 [min_x_case, max_x_case, min_y_case, max_y_case]
            box = data['box']  # 当前切片的感兴趣区域框，假设为 [min_x, max_x, min_y, max_y,]

            if self.crop_train == 'max_crop':
                x_start, y_start, x_end, y_end = self.max_crop(image_V, case_box, self.img_size)
            elif self.crop_train == 'random_crop':
                x_start, y_start, x_end, y_end = self.random_crop(image_V, box, case_box, self.img_size)
            else:
                x_start, y_start, x_end, y_end = 0, 0, image_V.shape[0], image_V.shape[1]

            # 对每个阶段的图像进行相同的裁剪
            image_V = image_V[x_start:x_end, y_start:y_end]
            image_A = image_A[x_start:x_end, y_start:y_end]
            image_D = image_D[x_start:x_end, y_start:y_end]
            label = label[x_start:x_end, y_start:y_end]

            # 调整图像和标签大小
            if image_V.shape[0] != self.img_size or image_V.shape[1] != self.img_size:
                image_V = zoom(image_V, (self.img_size / image_V.shape[0], self.img_size / image_V.shape[1]), order=3)
                image_A = zoom(image_A, (self.img_size / image_A.shape[0], self.img_size / image_A.shape[1]), order=3)
                image_D = zoom(image_D, (self.img_size / image_D.shape[0], self.img_size / image_D.shape[1]), order=3)
                label = zoom(label, (self.img_size / label.shape[0], self.img_size / label.shape[1]), order=0)

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.h5".format(vol_name)
            # data = h5py.File(filepath)
            # image, label = data['image_v'][:], data['label'][:]
            with h5py.File(filepath, 'r') as h5_file:
                keys = h5_file["box_keys"][:]
                values = h5_file["box_values"][:]
                crop_boxes = {key: value for key, value in zip(keys, values)}
                image_V, image_A, image_D, label, case_box = h5_file['image_V'][:], h5_file['image_A'][:], h5_file['image_D'][:], \
                 h5_file['label'][:], h5_file['case_box'][:]
            sample = {'image_V': image_V, 'image_A': image_A, 'image_D': image_D, 'label': label, 'case_box': case_box, 'box': crop_boxes}
            if self.norm_x_transform is not None:
                sample['image_V'] = self.norm_x_transform(sample['image_V'].copy())
                sample['image_A'] = self.norm_x_transform(sample['image_A'].copy())
                sample['image_D'] = self.norm_x_transform(sample['image_D'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
