import random
import os

import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.dataset as ds

class PairRandomCrop:
    """pair random crop"""
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, label):
        def _input_to_factor(img, size):
            """_input_to_factor"""
            img_height, img_width, _ = img.shape
            height, width = size
            if height > img_height or width > img_width:
                raise ValueError(f"Crop size {size} is larger than input image size {(img_height, img_width)}.")

            if width == img_width and height == img_height:
                return 0, 0, img_height, img_width

            top = random.randint(0, img_height - height)
            left = random.randint(0, img_width - width)
            return top, left, height, width

        y, x, h, w = _input_to_factor(image, self.size)
        image, label = image[y:y+h, x:x+w], label[y:y+h, x:x+w]
        assert image.shape == label.shape
        return image, label


class PairRandomHorizontalFlip:
    """pair random horizontal flip"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.prob:
            return img[::, ::-1], label[::, ::-1]
        return img, label


class PairRandomRGBShuffle:
    """pair random RGB shuffle"""
    def __init__(self):
        pass

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be RGB shuffled.

        Returns:
            PIL Image: Randomly RGB shuffled image.
        """
        perm = [0,1,2]
        random.shuffle(perm)
        img, label = np.array(img)[:, :, perm], np.array(label)[:, :, perm]
        return img, label
    

class DeblurDatasetGenerator:
    """DeblurDatasetGenerator"""
    def __init__(self, image_dir, do_train=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.random_rgb_shuffle = PairRandomRGBShuffle()
        self.random_horizontal_flip = PairRandomHorizontalFlip()
        self.random_crop = PairRandomCrop()
        self.do_train = do_train

    def __len__(self):
        """get len"""
        return len(self.image_list)

    def __getitem__(self, idx):
        """get item"""
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))
        image, label = image.convert("RGB"), label.convert("RGB")
        image, label = np.asarray(image), np.asarray(label)

        if self.do_train:
            image, label = self.random_crop(image, label)
            image, label = self.random_horizontal_flip(image, label)
            image, label = self.random_rgb_shuffle(image, label)
            
        image = image.astype(np.float32) / 255
        label = label.astype(np.float32) / 255

        image = image.transpose(2, 0, 1)  # transform to chw format
        label = label.transpose(2, 0, 1)  # transform to chw format

        return image, label

    @staticmethod
    def _check_image(lst):
        """check image format"""
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError(f"{x} is not .png, .jpeg or .jpg image")


def create_dataset(dataset_path, do_train, config, drop_remainder=True):
    """prepare dataset"""
    dataset_generator = DeblurDatasetGenerator(image_dir=dataset_path, do_train=do_train)
    # args.train_dataset_len = len(train_dataset_generator)
    dataset = ds.GeneratorDataset(source=dataset_generator, column_names=["image", "label"],
                                  shuffle=do_train, num_parallel_workers=config.num_parallel_workers,
                                  num_shards=config.rank_size, shard_id=config.rank_id)

    dataset = dataset.batch(batch_size=config.batch_size, drop_remainder=drop_remainder)
    return dataset