"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
from sklearn.neighbors import KDTree


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  def T(self):
    """End time of the SDE."""
    return 1

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas.to(t.device)[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G


def all_pairs_knn(points, k):
    points_np = points.flatten(start_dim=1).detach().cpu().numpy()
    tree = KDTree(points_np)
    _, indices = tree.query(points_np, k+1)
    indices = indices[:,1:]
    return points[indices]

def all_pairs_knn_torch(points, k):
    points_flattened = points.flatten(start_dim=1)
    distance_matrix = torch.cdist(points_flattened, points_flattened, p=2)
    _, indices = torch.topk(distance_matrix, k + 1, dim=0, largest = False)
    indices = indices[1:].T
    return points[indices]

class LaplacianVPSDE(VPSDE):
  def __init__(self, N, beta_min, beta_max, lmbda, k, eps=1e-3):
    super().__init__(beta_min=beta_min, beta_max=beta_max, N=N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.lmbda = lmbda
    self.eps = eps
    self.k = k

  def compute_laplacians(self, points):
    # neighbors = all_pairs_knn(points, self.k)
    neighbors = all_pairs_knn_torch(points, self.k)
    # TODO: try other weights, e.g., cotangent
    weights = torch.ones_like(neighbors)
    laplacians = 1 / self.k * torch.sum(weights*(neighbors - points[:,None]), dim=1)
    return laplacians

  def sde(self, x, t):
    drift, diffusion = super().sde(x, t)
    if self.lmbda != 0:
      laplacians = self.compute_laplacians(x)
      drift = drift - self.lmbda*laplacians
      # laplacians = laplacians/laplacians.norm(p=2, dim=(1,2,3), keepdim=True)
      # drift = drift - (1-t.reshape(-1, 1, 1, 1))*self.lmbda*laplacians
      # s_fun = lambda x, beta: 1 / (1 + (x/(1-x))**(-beta))
      # drift = drift - s_fun(1-t.reshape(-1, 1, 1, 1), 2)*self.lmbda*laplacians
      # drift = drift - self.lmbda*torch.einsum('i...,i...->i...', t, laplacians)
    return drift, diffusion

  def discretize(self, x, t):
    raise NotImplementedError()

  def forward_steps(self, x, t):
    # euler_maruyama update
    def update_fn(self, x, t):
      dt = 1. / self.N
      z = torch.randn_like(x)
      drift, diffusion = self.sde(x, t)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None, None] * np.sqrt(dt) * z
      return x, x_mean

    timesteps = torch.linspace(self.eps, self.T, self.N, device=t.device)
    x_t = torch.zeros_like(x)
    prev_t = 0
    for ts in timesteps:
      vec_t = torch.ones(x.shape[0], device=t.device) * ts
      x, _ = update_fn(self, x, vec_t)
      sel = torch.logical_and(t > prev_t, t <= ts)
      x_t[sel] = x[sel]
      prev_t = ts

    return x_t


class LaplacianVESDE(VESDE):
  def __init__(self, N, sigma_min, sigma_max, lmbda, k, eps=1e-3):
    super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, N=N)
    self.lmbda = lmbda
    self.eps = eps
    self.k = k

  def compute_laplacians(self, points):
    neighbors = all_pairs_knn_torch(points, self.k)
    # TODO: try other weights, e.g., cotangent
    weights = torch.ones_like(neighbors)
    laplacians = 1 / self.k * torch.sum(weights*(neighbors - points[:,None]), dim=1)
    return laplacians

  def sde(self, x, t):
    drift, diffusion = super().sde(x, t)
    if self.lmbda != 0:
      laplacians = self.compute_laplacians(x)
      drift = drift - self.lmbda*laplacians
    return drift, diffusion

  def discretize(self, x, t):
    raise NotImplementedError()

  def forward_steps(self, x, t):
    # euler_maruyama update
    def update_fn(self, x, t):
      dt = 1. / self.N
      z = torch.randn_like(x)
      drift, diffusion = self.sde(x, t)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None, None] * np.sqrt(dt) * z
      return x, x_mean

    timesteps = torch.linspace(self.eps, self.T, self.N, device=t.device)
    x_t = torch.zeros_like(x)
    prev_t = 0
    for ts in timesteps:
      vec_t = torch.ones(x.shape[0], device=t.device) * ts
      x, _ = update_fn(self, x, vec_t)
      sel = torch.logical_and(t > prev_t, t <= ts)
      x_t[sel] = x[sel]
      prev_t = ts

    return x_t
