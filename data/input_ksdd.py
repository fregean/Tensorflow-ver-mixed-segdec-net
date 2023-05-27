import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config


def read_split(train_num: int, num_segmented: int, fold: int, kind: str):
    fn = f"KSDD/split_{train_num}_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples[fold]
        elif kind == 'TEST':
            return test_samples[fold]
        else:
            raise Exception('Unknown')


class KSDDDataset(Dataset):

    def __init__(self, path: str, cfg: Config, kind: str):
        super(KSDDDataset, self).__init__(path, cfg, kind)
        self.read_contents()
        self.length = len(self._data)

    # @classmethod
    # def new(cls, path: str, cfg: Config, kind: str):
    #     instance = super(KSDDDataset, cls).__new__(cls, path, cfg, kind)
    #     instance.__init__(path, cfg, kind)
    #     return instance

    @classmethod
    def new(cls, path: str, cfg: Config, kind: str):
        return super(KSDDDataset, cls)._new(path, cfg, kind)

    @property
    def dataset_length(self):
        return self.length

    def read_contents(self):
        pos_samples, neg_samples = [], []

        folders = read_split(self.cfg.TRAIN_NUM, self.cfg.NUM_SEGMENTED, self.cfg.FOLD, self.kind)
        for f, is_segmented in folders:
            for sample in sorted(os.listdir(os.path.join(self.path, f))):
                if not sample.__contains__('label'):
                    image_path = self.path + '/' + f + '/' + sample
                    seg_mask_path = f"{image_path[:-4]}_label.bmp"
                    image = self.read_img_resize(image_path, self.grayscale, self.image_size)
                    seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, dilate=self.cfg.DILATE)
                    sample_name = f"{f}_{sample}"[:-4]
                    if sample_name == 'kos21_Part7':
                        continue
                    if positive:
                        image = self.to_tensor(image)
                        seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                        seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                        seg_mask = self.to_tensor(self.downsize(seg_mask))
                        pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, sample_name))
                    else:
                        image = self.to_tensor(image)
                        seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                        seg_mask = self.to_tensor(self.downsize(seg_mask))
                        neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, sample_name))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)
        # _data属性を設定
        self._data = pos_samples + neg_samples

        self.init_extra()
