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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE, LaplacianVPSDE, LaplacianVESDE


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, loss_type='dsm'):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def dsm_loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    mean, std = sde.marginal_prob(batch, t)

    z = torch.randn_like(batch)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)
    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  def ssm_loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    mean, std = sde.marginal_prob(batch, t)

    anneal_power = 2.0
    n_particles = 1

    if isinstance(sde, LaplacianVPSDE) or isinstance(sde, LaplacianVESDE):
      perturbed_samples = sde.forward_steps(batch, t)
    else:
      perturbed_samples = mean + torch.randn_like(batch) * std[:, None, None, None]

    if not train:
      torch.set_grad_enabled(True)
    dup_samples = (
        perturbed_samples.unsqueeze(0)
        .expand(n_particles, *batch.shape)
        .contiguous()
        .view(-1, *batch.shape[1:])
    )
    dup_labels = t.unsqueeze(0).expand(n_particles, *t.shape).contiguous().view(-1)
    dup_samples.requires_grad_(True)

    # use Rademacher
    vectors = torch.randn_like(dup_samples)

    dup_samples.requires_grad_()
    grad1 = score_fn(dup_samples, dup_labels)
    gradv = torch.sum(grad1 * vectors)
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    if not train:
      torch.set_grad_enabled(False)

    grad1 = grad1.view(dup_samples.shape[0], -1)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0

    loss2 = torch.sum((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = (loss1 + loss2) * (std.squeeze() ** anneal_power)

    return loss.mean(dim=0)

  def esm_loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    mean, std = sde.marginal_prob(batch, t)

    n_particles = 1
    anneal_power = 2.0

    if isinstance(sde, LaplacianVPSDE) or isinstance(sde, LaplacianVESDE):
      perturbed_samples = sde.forward_steps(batch, t)
    else:
      perturbed_samples = mean + torch.randn_like(batch) * std[:, None, None, None]

    data_dim = np.prod(batch.shape[1:])
    dup_samples = (
        perturbed_samples.unsqueeze(0)
        .expand(n_particles, *batch.shape)
        .contiguous()
        .view(-1, *batch.shape[1:])
    )
    dup_labels = t.unsqueeze(0).expand(n_particles, *t.shape).contiguous().view(-1)
    dup_samples.requires_grad_(True)

    # use Rademacher
    eps_fd = 0.1
    vectors = torch.randn_like(dup_samples)
    vectors = (
        vectors / torch.sqrt(torch.sum(vectors ** 2, dim=[1, 2, 3], keepdim=True)) * eps_fd
    )
    # vectors = vectors / torch.sqrt(torch.sum(vectors**2, dim=[1,2,3], keepdim=True)) * 10. * used_sigmas

    cat_input = torch.cat([dup_samples + vectors, dup_samples - vectors], 0)
    cat_label = torch.cat([dup_labels, dup_labels], 0)

    grad_all = score_fn(cat_input, cat_label)

    batch_size = dup_samples.shape[0]
    grad_all = grad_all.view(2 * batch_size, -1)
    vectors = vectors.view(batch_size, -1)
    out_1 = grad_all[:batch_size]
    out_2 = grad_all[batch_size:]

    grad1 = out_1 + out_2
    grad2 = out_1 - out_2

    loss_1 = (grad1 * grad1) / 8.0
    loss_2 = grad2 * vectors * (data_dim / (2.0 * eps_fd * eps_fd))
    loss = torch.sum(loss_1 + loss_2, dim=-1)

    loss = loss * (std.squeeze() ** anneal_power)

    return loss.mean(dim=0)

  if loss_type == 'dsm':
    return dsm_loss_fn
  elif loss_type == 'ssm':
    assert not likelihood_weighting, 'Likelihood weighting implemented only for dsm'
    return ssm_loss_fn
  elif loss_type == 'esm':
    assert not likelihood_weighting, 'Likelihood weighting implemented only for dsm'
    return esm_loss_fn
  else:
    raise RuntimeError(f'Loss function not recognised: {loss_type}')


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, loss_type='dsm', continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
      loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, loss_type=loss_type,
                                continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
