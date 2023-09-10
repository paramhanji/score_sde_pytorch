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
  training.sde = 'laplacian'
  training.continuous = True
  training.reduce_mean = True
  training.batch_size = 1024
  training.n_iters = 10000
  training.snapshot_freq = 1000
  training.log_freq = 25
  training.eval_freq = 1000

  # optimizer
  optim = config.optim
  optim.lr = 1e-3
  optim.warmup = 0

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.store_intermediate = True
  sampling.n_steps_each = 1
  sampling.noise_removal = False
  sampling.probability_flow = False
  sampling.snr = 0.16

  # data
  data = config.data
  data.centered = False
  data.dataset = 'Toy2D'
  data.train_samples = 15000
  data.eval_samples = 4096
  data.num_channels = 2
  data.image_size = 1
  data.noise = 0.05

  # model
  model = config.model
  model.name = 'ddpm_small'
  model.ema_rate = 0.9
  model.conditional = True
  model.beta_max = 5.
  model.num_scales = 50
  model.lap_lambda = 1.
  model.lap_k = 10

  # eval
  eval = config.eval
  eval.save_to_file = False
  eval.batch_size = 1024
  eval.enable_loss = True
  eval.enable_bpd = True
  eval.bpd_dataset = 'test'
  eval.enable_sampling = False
  eval.begin_ckpt = 1
  eval.end_ckpt = 10
  eval.enable_stats = True

  return config
