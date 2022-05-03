from mmdet.models import Shared2FCBBoxHead
import torch.nn as nn
import torch


class DisentangleLinear(nn.Module):
    def __init__(self, in_features, eigen_vectors):
        super(DisentangleLinear, self).__init__()
        self.in_features = in_features
        self.eigen_vectors = eigen_vectors
        self.W = nn.Parameter(torch.eye(in_features))
        self.b = nn.Parameter(torch.zeros(in_features, 1))

    def _get_eigen_vectors(self):
        U, S, Vh = torch.linalg.svd(self.W.detach())
        # For stability in training, no gradient in svd

        return Vh[-self.eigen_vectors:]

    def forward(self, x):
        x = x.T + self.b                   # CxN
        vh = self._get_eigen_vectors()     # ExC
        x_null = vh.T @ (vh @ x)           # CxN
        x = x - x_null

        return (self.W @ x).T.contiguous()   # , x_null.T.contiguous()


class CustomeShared2FCBBoxHead(Shared2FCBBoxHead):
    def __init__(self, num_nulls, *args, **kwargs):
        super(CustomeShared2FCBBoxHead, self).__init__(*args, **kwargs)
        assert self.with_cls
        if self.custom_cls_channels:
            cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
        else:
            cls_channels = self.num_classes + 1
        disentangle_layer = DisentangleLinear(in_features=cls_channels,
                                            eigen_vectors=num_nulls)
        self.fc_cls = nn.Sequential(self.fc_cls,
                                    disentangle_layer)
