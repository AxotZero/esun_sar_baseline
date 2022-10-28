from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from .modules.feature_embedder import FeatureEmbedder
from .modules.temporal_aggregator import *
from process_data.utils import load_yaml
from process_data.data_config import DATA_SOURCES


class SarModel(BaseModel):
    def __init__(self,
                 num_cat_pkl_path="", emb_feat_dim=32, hidden_size=128, hidden_size_coeff=6, dropout=0.3,
                 max_len=512,
                 temporal_aggregator_type="TemporalTransformerAggregator", temporal_aggregator_args={},
                 ):
        super().__init__()
        num_cat_dict = load_yaml(num_cat_pkl_path)
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.feature_embedders = nn.ModuleList([
            FeatureEmbedder(num_cat_dict, ds, emb_feat_dim, hidden_size, hidden_size_coeff, dropout) 
            for ds in DATA_SOURCES
        ])
        
        self.temporal_aggregator = eval(
            f"{temporal_aggregator_type}")(**temporal_aggregator_args)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_aggregator_args["hidden_size"], 1),
            nn.Sigmoid()
        )

    def forward(self, batch_idxs, seq_idxs, xs):
        """
        batch_idxs: batch_idxs of 5 data_source, shape(5, len(given_data_source))
        seq_idxs: seq_idxs of 5 data_source, shape(5, len(given_data_source))
        xs: a list of data of given data_source, shape(5, len(given_data_source), given_data_source_features)
        """

        ## embed feature
        batch_size = int(max([max(b) for b in batch_idxs])) + 1
        xs = [self.feature_embedders[i](x) for i, x in enumerate(xs)]
        # create zero embedding map
        _x = torch.zeros((batch_size, self.max_len, self.hidden_size)).to(xs[0].get_device()).float()
        mask = torch.zeros((batch_size, self.max_len)).to(xs[0].get_device()).long()
        # put features to the right position and set the mask
        for bi, si, x in zip(batch_idxs, seq_idxs, xs):
            _x[bi, si] = x
            mask[bi, si] = 1

        # run temporal model
        x = self.temporal_aggregator(_x, mask)
        x = self.classifier(x).squeeze(-1)
        return x


if __name__ == "__main__":
    pass
    # model = SelfAttenNN('../data/preprocessed/v1/column_config_generated.yml')
    # data = torch.zeros((32, 200, 52))
    # print(model(data))
