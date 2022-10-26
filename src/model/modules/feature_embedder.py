from operator import ge
from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F
from process_data.data_config import CONFIG_MAP, FeatureType
from process_data.utils import get_feats_name


class CnnEncoder(nn.Module):
    """
    src: https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
    """
    def __init__(self, num_features, num_targets=128, hidden_size=512, dropout=0.3):
        super().__init__()
        cha_1 = 64
        cha_2 = 128
        cha_3 = 128

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(dropout*0.9)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(dropout*0.8)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(dropout*0.6)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(dropout*0.5)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()
        
        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                        self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class FeatureEmbedder(torch.nn.Module):
    """
    Classical embeddings generator
    src: tabnet
    """

    def __init__(self, num_cat_dict, data_source, emb_feat_dim=32, hidden_size=128, hidden_size_coeff=6, dropout=0.2):
        """This is an embedding module for an entire set of features
        Parameters
        ----------
        """
        super().__init__()
        feats_name = get_feats_name(CONFIG_MAP[data_source])
        feats_type = [getattr(CONFIG_MAP[data_source], name) for name in feats_name]
        self.feats_type = feats_type

        self.source_type_embedding = nn.Parameter(torch.randn(hidden_size))
        self.embeddings = torch.nn.ModuleList(
            [   
                nn.Linear(1, emb_feat_dim) if feat_type == FeatureType.NUMERICAL
                else nn.Embedding(num_cat_dict[data_source][feat_name], emb_feat_dim)
                for feat_name, feat_type in zip(feats_name, feats_type)
            ]
        )
        self.encoder = CnnEncoder(
            num_features=len(feats_name)*emb_feat_dim, 
            num_targets=hidden_size, 
            hidden_size=hidden_size*hidden_size_coeff,
            dropout=dropout
        )


    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        embs = []
        for i, (feat_type, emb_layer)  in enumerate(zip(self.feats_type, self.embeddings)):
            # Enumerate through continuous idx boolean mask to apply embeddings
            inputs = x[:, i]
            if feat_type == FeatureType.NUMERICAL:
                inputs = inputs.view(-1, 1).float()
            else:
                inputs = inputs.long()
            embs.append(emb_layer(inputs))
        embs = torch.cat(embs, dim=1)
        embs = self.encoder(embs)
        embs += self.source_type_embedding
        return embs