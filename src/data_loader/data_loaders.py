from ast import get_source_segment
from os import device_encoding
from pdb import set_trace as bp


from easy_dict import EasyDict as edict

import torch
from torch.utils.data import Dataset

from base import BaseDataLoader
from process_data.data_config import CONFIG_MAP, DATA_SOURCES, 
from process_data.utils import load_pickle, get_feats_name

# def padding_mask_collate(batch):
#     """
#     return 
#     """


def batch_index_collate(data):
    data = list(zip(*data))
    y = torch.stack(data[1], 0)
    
    batch_indices = []
    for i, d in enumerate(data[0]):
        batch_indices += [i] * int(d.size()[0])
        
    return (
        (torch.Tensor(batch_indices), torch.cat(data[0]).float()), 
        y.float()
    )


class MaxLenDataLoader(BaseDataLoader):
    class InnerDataset(Dataset):
        def __init__(self, data_path, max_len=512, training=True):
            self.training = training
            self.max_len = max_len
            self.load_xs(data_path)

        def load_xs(self, data_path):
            print('loading data')
            pkl = load_pickle(data_path)
            self.data = []
            for k, v in pkl.items():
                masks = v.train_mask if self.training else v.test_mask
                for e in masks:
                    e += 1
                    s = max(e - self.max_len, 0)
                    self.data.append(edict({
                        'sources': v.sources[s:e],
                        'cust_data': v.cust_data[s:e]
                    }))
            print(f'num of data: {len(self.data)}')

        def __len__(self):
            return len(self.data)

        def get_source_data(self, datas, sources, source_type):
            ret = []
            config = CONFIG_MAP[source_type]
            feats_name = get_feats_name(config)
            for seq_idx, (source, data) in enumerate(zip(sources, datas)):
                if source != source_type:
                    continue
                d = [data[feat_name] for feat_name in feats_name]
                ret.append((seq_idx, d))
            return ret
                
        def __getitem__(self, i):
            data = self.data[i]
            sources = data.soucres
            cust_data = data.cust_data

            x = (self.get_source_data(cust_data, sources, ds) for ds in DATA_SOURCES)

            if self.training:
                y = cust_data[-1].sar_flag
            else:
                y = cust_data[-1].alert_key
            return (x, y)
            

    class BatchCollect:
        def __init__(self, max_len=512, training=True, device=torch.device('cuda')):
            self.max_len = max_len
            self.training = training
            self.device = device

        def __call__(self, datas):
            xs, ys = list(zip(*datas))
            batch_idxs = [[] for i in range(len(xs[0]))]
            seq_idxs = [[] for i in range(len(xs[0]))]
            ret_xs = [[] for i in range(len(xs[0]))]

            for batch_idx, x in enumerate(xs):
                # add batch idx
                for i, (seq_idx, v) in enumerate(x):
                    batch_idxs[i] += [batch_idx] * len(seq_idx)
                    seq_idxs[i] += seq_idx
                    ret_xs[i] += v
                    
            if self.training:
                ys = torch.tensor(ys).float().to(self.device)
            return (
                (torch.tensor(b).long().to(self.device) for b in batch_idxs),
                (torch.tensor(s).long().to(self.device) for s in seq_idxs), 
                (torch.tensor(x).float().to(self.device) for x in ret_xs),  # (ccba, cdtx, dp, remit, cinfo), 
                ys
            )
            
    
    def __init__(self, 
                 data_path, max_len=512, device=torch.device('cuda'),
                 batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.0, num_workers=1, training=True):
        cls = self.__class__
        self.dataset = cls.InnerDataset(data_path, training=training)
        super().__init__(
            self.dataset, 
            batch_size, shuffle, fold_idx, validation_split, num_workers, 
            collate_fn=cls.BatchCollate(max_len, training, device)
        )