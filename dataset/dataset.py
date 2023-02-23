import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from dataset.augments import rotate, blur, get_number_bb, get_contour_cutout

class Dataset:
    def __init__(self, directory, df, label_col, config, mode='train'):
        '''
        directory = parent directory of the dataset
        '''
        self.directory = directory
        self.df = df
        self.label_col = label_col
        self.mode = mode
        self.config = config

        self.cache = self.config['cache']
        self.cache_data = {}
        self.cache_contour = {}
        self.cache_bbox = {}
        if self.cache: self.cache_image()

    def __len__(self):
        return len(self.df)
    
    def change_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # opening
        if self.config['opening']: image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        
        # use only bounding box
        if self.config['use_bbox'] or (self.config['contour_cutout_prob'] > 0.0): 
            bounding_box, contours = get_number_bb(image, self.config['use_bbox'])
            self.cache_contour[path] = contours
            self.cache_bbox[path] = bounding_box
        # reverse
        if self.config['reverse']: image = 255 - image
        # dilation
        if self.config['dilation']: image = cv2.dilate(image, np.ones((5, 5), np.uint8), iterations = 1)
        
        # resize
        # if not train need to apply these now as no contour cutout in valid/test
        if self.mode != 'train' or (self.config['contour_cutout_prob'] == 0.0):
            if self.config['use_bbox']:
                bounding_box = self.cache_bbox[path]
                image = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], :]

            # need to apply contour before resize so this condition is applied
            image = cv2.resize(image, (self.config['img_shape'][0], self.config['img_shape'][1]), interpolation = cv2.INTER_AREA)

        return image
    
    def cache_image(self):
        print("Cache Dataset...")
        for i in tqdm(range(len(self.df))):
            row = self.df.iloc[i]
            path = os.path.join(self.directory, row['img_path'])
            self.cache_data[path] = self.change_image(path)

    def augment(self, image):
        if np.random.rand() < 0.5:
            image = rotate(image, -10, 10)
        if np.random.rand() < 0.5:
            image = blur(image)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.directory, row['img_path'])

        if self.cache:
            image = self.cache_data[path]
        else:
            image = self.change_image(path)

        # use probabilistic augmentation
        if self.config['aug'] and self.mode == 'train': image = self.augment(image)

        # use contour cutout
        if self.mode == 'train':
            # need to apply contour before resize and/or bbox so this condition is applied
            if (self.config['contour_cutout_prob'] > np.random.rand()):
                image = get_contour_cutout(image, self.cache_contour[path], self.config['contour_cutout_number'])
            if self.config['use_bbox']:
                bounding_box = self.cache_bbox[path]
                image = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], :]
            image = cv2.resize(image, (self.config['img_shape'][0], self.config['img_shape'][1]), interpolation = cv2.INTER_AREA)
            
        # resize and normalize
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        
        if self.mode in ["train", "valid"] : return image, row[self.label_col]
        else: return image, row['img_path']


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.idx = 0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            print("Shuffling Dataset. ")
            self.dataset.df = self.dataset.df.sample(frac=1).reset_index(drop=True)
        return self

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration


        batch = []       
        for i in range(self.batch_size):
            if self.idx >= len(self.dataset):
                break
            data = self.dataset[self.idx]
            if self.dataset.mode in ["train", "valid"]: 
                if len(batch) == 0: batch = [[], []]
                batch[0].append(data[0])
                batch[1].append(data[1])
            else: 
                if len(batch) == 0: batch = [[], []]
                batch[0].append(data[0])
                batch[1].append(data[1])
            self.idx += 1

        if self.dataset.mode in ["train", "valid"]: batch = [np.stack(batch[0]), np.array(batch[1])]
        else: batch = [np.stack(batch[0]), batch[1]]
        return batch


    
def check_dataset(train_dataset, valid_dataset, save_dir, from_mixup=False):
    train_idx = np.random.randint(0, len(train_dataset))
    valid_idx = np.random.randint(0, len(valid_dataset))

    if from_mixup:
        train_image = train_dataset[0][train_idx]
        train_label = train_dataset[1][train_idx]
    else:
        train_image, train_label = train_dataset[train_idx]
    valid_image, valid_label = valid_dataset[valid_idx]


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(train_image.transpose(1, 2, 0))
    ax[0].set_title(f"Train[{train_idx}]: {train_label}")
    ax[1].imshow(valid_image.transpose(1, 2, 0))
    ax[1].set_title(f"Valid[{valid_idx}]: {valid_label}")

    # save as image
    fig.savefig(f'{save_dir}/dataset.png', dpi=300, bbox_inches='tight')
    # plt.show()

def check_test_dataset(test_dataset, save_dir):
    test_idx = np.random.randint(0, len(test_dataset))
    test_image, path = test_dataset[test_idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(test_image.transpose(1, 2, 0))
    ax.set_title(f"Test[{test_idx}]")

    # save as image
    fig.savefig(f'{save_dir}/dataset_test.png', dpi=300, bbox_inches='tight')
    # plt.show()



