"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch.nn as nn
import torch
from losses.am_softmax import AngleSimpleLinear
from .common import ModelInterface

class ResNetAngularKP(ModelInterface):
    """Face reid head for the ResNet architecture"""
    def __init__(self, backbone, embedding_size=128, num_classes=0, feature=True):
        super(ResNetAngularKP, self).__init__()

        self.bn_first = nn.InstanceNorm2d(3, affine=True)
        self.feature = feature
        self.model = backbone 
        self.embedding_size = embedding_size
        self.kp = nn.Linear(136, 68)

        if not self.feature:
            self.fc_angular = AngleSimpleLinear(self.embedding_size+68, num_classes)

    def forward(self, x, kp=None):

        x = self.bn_first(x)
        x = self.model(x)
        
        if kp is not None:
            kp = kp.view(x.shape[0], -1).float()
#             print(kp.shape)
            kpf = self.kp(kp)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        
        x = torch.cat([x,kpf],-1)
#         print(x.shape)
        y = self.fc_angular(x)

        return x, y

    @staticmethod
    def get_input_res():
        return (None, None)

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.
        
        
class ResNetAngular(ModelInterface):
    """Face reid head for the ResNet architecture"""
    def __init__(self, backbone, embedding_size=128, num_classes=0, feature=True):
        super(ResNetAngular, self).__init__()

#         self.bn_first = nn.InstanceNorm2d(3, affine=True)
        self.feature = feature
        self.model = backbone 
        self.embedding_size = embedding_size
        self.dropout=nn.Dropout(0.1)

        if not self.feature:
            self.fc_angular = AngleSimpleLinear(self.embedding_size, num_classes)

    def forward(self, x):

#         x = self.bn_first(x)
        x = self.model(x)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        y = self.fc_angular(x)

        return x, y

    @staticmethod
    def get_input_res():
        return (None, None)

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.