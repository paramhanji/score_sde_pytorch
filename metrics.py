import torch
from torch_two_sample import MMDStatistic, FRStatistic, EnergyStatistic
from itertools import tee
from functools import partial
from tqdm import trange

def compute_bpd(ds, scaler, likelihood_fn, score_model, bpd_num_repeats, device='cuda'):
  bpds = []
  for _ in trange(bpd_num_repeats, leave=False):
    bpd_iter = iter(ds)  # pytype: disable=wrong-arg-types
    for _ in trange(len(ds), leave=False):
      batch = next(bpd_iter)
      eval_batch = torch.from_numpy(batch['image']._numpy()).to(device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      bpd = likelihood_fn(score_model, eval_batch)[0]
      bpd = bpd.detach().cpu().numpy().reshape(-1)
      bpds.extend(bpd)

  return bpds


def compute_stats(ds, scaler, sampling_fn, score_model, store_intermediate, device='cuda'):

    dummy_ds, ds = tee(ds, 2)
    batch_size = next(dummy_ds)['image'].shape[0]
    tests = {'mmd':partial(MMDStatistic(batch_size, batch_size), alphas=[0.1]),
             'fr':FRStatistic(batch_size, batch_size),
             'energy':EnergyStatistic(batch_size, batch_size)}
    results = {'mmd':[], 'fr':[], 'energy':[]}
    eval_iter = iter(ds)  # pytype: disable=wrong-arg-types
    for batch in eval_iter:
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch).reshape(batch_size, -1)
        with torch.no_grad():
            generated_batch, _ = sampling_fn(score_model)
            if store_intermediate:
               generated_batch = generated_batch[-1]
            generated_batch = generated_batch.reshape(batch_size, -1)
        
        for test, fn in tests.items():
           results[test].append(fn(eval_batch, generated_batch).item())

    return results