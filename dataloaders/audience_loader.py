import os
from glob import glob
import numpy as np
import cv2
from torch.utils.data import Dataset
import skimage.io as skio

from conf import AUDIENCE_PATH


class AudienceDataset(Dataset):

    def __init__(
        self, split, seq_len, 
        img_side=64, dataset_dir=AUDIENCE_PATH, data_augmentation=True, normalize=True):

        self.split = split
        self.seq_len = seq_len
        self.img_side = img_side
        self.normalize = normalize

        data_dir = dataset_dir
        self.example_dirs = glob(os.path.join(data_dir, '*'))
        if split == 'train':
            self.example_dirs = self.example_dirs[0:int(.9*len(self.example_dirs))]
        else:
            self.example_dirs = self.example_dirs[int(.9*len(self.example_dirs)):]

        self.transforms = None

    def __len__(self):
        return len(self.example_dirs)
        
    def _load_frame(self, frame_path):

        # Load image
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to desired image shape
        # if self.img_side != 128:
        img = cv2.resize(img, (self.img_side, self.img_side))


        return img
        
    def __getitem__(self, item_idx):
        example_dir = self.example_dirs[item_idx]

        # List jpgs in dir
        frames = glob(os.path.join(example_dir, '*.jpg'))
        # Sort by frame idx
        frames = sorted(frames, key=lambda x: int(x.split('\\')[-1].split('.')[0]))

        # Return frames
        if self.seq_len is not None:
            if self.split == 'train':
                start_frame = np.random.randint(0, len(frames) - self.seq_len)
            else:
                start_frame = 0
            frames = frames[start_frame: start_frame + self.seq_len]

        frames = [self._load_frame(f) for f in frames]
        frames = np.array(frames).astype(np.float32)
        frames = frames/255.

        if self.transforms is not None:
            frames = self.transforms(frames)
        
        frames = frames.transpose((0, 3, 1, 2))

        return (frames, item_idx)
