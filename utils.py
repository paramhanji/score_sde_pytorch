import numpy as np
import pandas as pd
import plotly.express as px
import torch
import tensorflow as tf
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


def save_gif(points, filename, num_highlight=5):
    points = torch.stack(points).squeeze().detach().cpu().numpy()
    n_steps, n_samples = points.shape[:2]
    colors = np.zeros((n_steps, n_samples))
    colors[:,-num_highlight:] = np.arange(1, num_highlight+1)

    t = np.arange(n_steps).repeat(n_samples)
    df = pd.DataFrame({'x': points[...,0].flatten(), 'y': points[...,1].flatten(), 't': t, 'color': colors.flatten()})
    df.color = df.color.astype(str)

    plot = px.scatter(df, x='x', y='y', animation_frame='t', color='color')
    plot.update_layout(autosize=False, width=500, height=500)
    plot.write_html(filename)
