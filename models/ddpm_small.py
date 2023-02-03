# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ddpm_small')
class DDPMSmall(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.n_steps = config.model.num_scales
    w = 256
    t_w = 16
    self.t_layer = nn.Sequential(nn.Linear(1, 16),
                              nn.Tanh(),
                              nn.Linear(16, t_w),
                              nn.Tanh())
    self.layer1 = nn.Sequential(nn.Linear(t_w + 2, w),
                                nn.Tanh(),
                                nn.Linear(w, w),
                                nn.Tanh())

    self.layer2 = nn.Sequential(nn.Linear(w, w),
                                nn.Tanh(),
                                nn.Linear(w, w),
                                nn.Tanh())
    self.layer3 = nn.Sequential(nn.Linear(w + t_w, w),
                                nn.Tanh(),
                                nn.Linear(w, w),
                                nn.Tanh())

    self.last_layer = nn.Linear(w, 2)

  def forward(self, x, t):
      t = (t.unsqueeze(-1).float() / self.n_steps) - 0.5
      x = x.squeeze(-1).squeeze(-1)
      temb = self.t_layer(t)

      output = self.layer1(torch.concat([x, temb], axis=-1))
      output = self.layer2(output)
      output = self.layer3(torch.concat([output, temb], axis=-1))
      return self.last_layer(output).unsqueeze(-1).unsqueeze(-1)
