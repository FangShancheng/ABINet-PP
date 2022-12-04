import logging
import re
import torch
import pandas as pd
import numpy as np        
import random

from fvcore.common.timer import Timer
from pathlib import Path
from torch import tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset
from detectron2.utils.comm import get_world_size
from torch.utils.data.distributed import DistributedSampler


CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
CTLABELS_STR = ''.join(CTLABELS[1:])

logger = logging.getLogger(__name__)

def onehot(label, depth, device=None):
    """ 
    Args:
        label: shape (n1, n2, ..., )
        depth: a scalar

    Returns:
        onehot: (n1, n2, ..., depth)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
    onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

    return onehot

def build_train_loader_text(cfg, bs=None, one_hot_y=False, num_workers=None):
    path = 'datasets/WikiText-103-n96.csv'
    num_workers = num_workers or cfg.DATALOADER.NUM_WORKERS
    total_batch_size = bs or cfg.SOLVER.IMS_PER_BATCH
    use_sm = cfg.MODEL.ABINET.LANGUAGE_USE_SM
    max_length = cfg.MODEL.BATEXT.NUM_CHARS
    dataset = TextDataset(path=path,
                          max_length=max_length,
                          is_training=True,
                          one_hot_y=one_hot_y,
                          use_sm=use_sm)

    world_size = get_world_size()
    # assert (
    #     total_batch_size > 0 and total_batch_size % world_size == 0
    # ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
    #     total_batch_size, world_size
    # )
    batch_size = total_batch_size // world_size
    if world_size > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    dataloaer = DataLoader(dataset, batch_size, sampler=sampler, drop_last=True, num_workers=num_workers)
    return TextIterator(dataloaer)


class TextIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try: d = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            d = next(self.dataloader_iter)
        return d


class TextDataset(Dataset):
    def __init__(self,
                 path:str, 
                 delimiter:str='\t',
                 max_length:int=25,
                 case_sensitive=True, 
                 one_hot_x=True,
                 one_hot_y=False,
                 is_training=True,
                 smooth_label=False,
                 smooth_factor=0.2,
                 use_sm=False,
                 **kwargs):
        self.path = Path(path)
        self.case_sensitive, self.max_length = case_sensitive, max_length
        self.smooth_factor, self.smooth_label = smooth_factor, smooth_label
        self.one_hot_x, self.one_hot_y, self.is_training = one_hot_x, one_hot_y, is_training
        self.use_sm = use_sm

        dtype = {'inp': str, 'gt': str}

        timer = Timer()
        self.df = pd.read_csv(self.path, dtype=dtype, delimiter=delimiter, na_filter=False)
        if self.is_training and self.use_sm: self.sm = SpellingMutation()
        self.inp_col, self.gt_col = 0, 1
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(self.path, timer.seconds()))
        logger.info("Loaded {} items in COCO format from {}".format(len(self), self.path))


    def __len__(self): return len(self.df)

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """ Returns the labels of the corresponding text.
        """
        length = length if length else self.max_length
        if not case_sensitive:
            text = text.lower()
        labels = [CTLABELS.index(char) for char in text]
        if padding:
            labels = labels + [len(CTLABELS)+1] * (length - len(text))
        return labels

    def augment(self, text, aug_idx):
        if aug_idx == 0:
            return text.lower()
        elif aug_idx == 1:
            return text.upper()
        elif aug_idx == 2:
            return text.capitalize()
        else:
            raise IndexError('Error aug_index')
        

    def __getitem__(self, idx):
        aug_idx = np.random.randint(0,3)
        text_x = self.df.iloc[idx, self.inp_col]
        #text_x = re.sub('[^0-9a-zA-Z]+', '', text_x)
        text_x = re.sub(f'[^{CTLABELS_STR}]+', '', text_x)
        text_x = self.augment(text_x, aug_idx)

        if self.is_training and self.use_sm: text_x = self.sm(text_x)
        if not self.case_sensitive: text_x = text_x.lower()

        length_x = tensor(len(text_x) + 1).to(dtype=torch.long)  # one for end token
        label_x = self.get_labels(text_x, case_sensitive=self.case_sensitive)
        label_x = tensor(label_x)
        if self.one_hot_x:
            label_x = onehot(label_x, len(CTLABELS) + 2)
            if self.is_training and self.smooth_label: 
                label_x = torch.stack([self.prob_smooth_label(l) for l in label_x])
    
        text_y = self.df.iloc[idx, self.gt_col]
        #text_y = re.sub('[^0-9a-zA-Z]+', '', text_y)
        text_y = re.sub(f'[^{CTLABELS_STR}]+', '', text_y)
        text_y = self.augment(text_y, aug_idx)
        if not self.case_sensitive: text_y = text_y.lower()
        length_y = tensor(len(text_y) + 1).to(dtype=torch.long)  # one for end token
        label_y = self.get_labels(text_y, case_sensitive=self.case_sensitive)
        label_y = tensor(label_y)
        if self.one_hot_y: label_y = onehot(label_y, len(CTLABELS) + 2)

        return {'tokens': label_x, 'lengths': length_x, 'gt_instances': label_y}


    def prob_smooth_label(self, one_hot):
        one_hot = one_hot.float()
        delta = torch.rand([]) * self.smooth_factor
        num_classes = len(one_hot)
        noise = torch.rand(num_classes)
        noise = noise / noise.sum() * delta
        one_hot = one_hot * (1 - delta) + noise
        return one_hot

class SpellingMutation(object):
    def __init__(self, pn0=0.7, pn1=1.0, pn2=1.0, pt0=0.7, pt1=0.85):
        """ 
        Args:
            pn0: the prob of not modifying characters is (pn0)
            pn1: the prob of modifying one characters is (pn1 - pn0)
            pn2: the prob of modifying two characters is (pn2 - pn1), 
                 and three (1 - pn2)
            pt0: the prob of replacing operation is pt0.
            pt1: the prob of inserting operation is (pt1 - pt0),
                 and deleting operation is (1 - pt1)
        """
        super().__init__()
        self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        self.pt0, self.pt1 = pt0, pt1
        logger.info(f'the probs: pn0={self.pn0}, pn1={self.pn1} ' + 
                     f'pn2={self.pn2}, pt0={self.pt0}, pt1={self.pt1}')

    def is_digit(self, text, ratio=0.5):
        length = max(len(text), 1)
        digit_num = sum([t in self.digits for t in text])
        if digit_num / length < ratio: return False
        return True

    def is_unk_char(self, char):
        return (char not in self.digits) and (char not in self.alphabets)

    @property
    def digits(self):
        return '0123456789'

    @property
    def alphabets(self):
        return 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def get_num_to_modify(self, length):
        prob = random.random()
        if prob < self.pn0: num_to_modify = 0
        elif prob < self.pn1: num_to_modify = 1
        elif prob < self.pn2: num_to_modify = 2
        else: num_to_modify = 3
        
        if length <= 1: num_to_modify = 0
        elif length >= 2 and length <= 4: num_to_modify = min(num_to_modify, 1)
        else: num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def __call__(self, text, debug=False):
        if self.is_digit(text): return text
        length = len(text)
        num_to_modify = self.get_num_to_modify(length)
        if num_to_modify <= 0: return text

        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]
        if debug: self.index = index
        for i, t in enumerate(text):
            if i not in index: chars.append(t)
            elif self.is_unk_char(t): chars.append(t)
            else:
                prob = random.random()
                if prob < self.pt0: # replace
                    chars.append(random.choice(self.alphabets))
                elif prob < self.pt1: # insert
                    chars.append(random.choice(self.alphabets))
                    chars.append(t)
                else: # delete
                    continue
        new_text = ''.join(chars)
        # new_text = ''.join(chars[: max_length-1])
        return new_text if len(new_text) >= 1 else text
