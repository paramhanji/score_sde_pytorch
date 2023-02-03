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

# Lint as: python3
"""Config file for reproducing the results of DDPM on cifar-10."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True
  training.batch_size = 1024
  training.n_iters = 50000
  training.snapshot_freq = 5000
  training.log_freq = 100
  training.eval_freq = 1000

  # optimizer
  optim = config.optim
  optim.lr = 1e-4
  optim.warmup = 0
  optim.grad_clip = 1e6

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'
  sampling.store_intermediate = True

  # data
  data = config.data
  data.centered = False
  data.dataset = 'Toy2D'
  data.train_samples = 15000
  data.eval_samples = 512
  data.num_channels = 2
  data.image_size = 1

  # model
  model = config.model
  model.name = 'ddpm_small'
  model.ema_rate = 0.9
  # model.scale_by_sigma = False
  # model.normalization = 'GroupNorm'
  # model.nonlinearity = 'swish'
  # model.nf = 1
  # model.ch_mult = (1, 2, 2, 2)
  # model.num_res_blocks = 2
  # model.attn_resolutions = (16,)
  # model.resamp_with_conv = True
  model.conditional = True
  model.beta_min = 1e-4
  model.beta_max = 2e-2
  model.num_scales = 100

  # eval
  eval = config.eval
  eval.batch_size = 4096

  return config
