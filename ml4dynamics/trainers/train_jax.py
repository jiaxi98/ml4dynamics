import argparse
import os
import pickle
from functools import partial
from time import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from flax import traverse_util
from matplotlib import pyplot as plt
from tqdm import tqdm

from ml4dynamics.utils import utils


def main():

  def train(
    mode: str,
    epochs: int,
  ):

    @partial(jax.jit, static_argnums=(3, ))
    def train_step(x, y, train_state, is_training=True):

      def loss_fn(params, batch_stats, is_training):
        if _global:
          y_pred, batch_stats = train_state.apply_fn_with_bn(
            {
              "params": params,
              "batch_stats": batch_stats
            },
            x,
            is_training=is_training
          )
        else:
          y_pred = train_state.apply_fn(params, x.reshape(-1,
                                                          x.shape[-1])).reshape(
                                                            y.shape
                                                          )
          batch_stats = None
        loss = jnp.mean((y - y_pred)**2)

        if mode == "ae":
          loss = jnp.mean((x - y_pred)**2)
        elif mode == "mols":
          squared_norms = jax.tree_map(lambda param: jnp.sum(param**2), params)
          loss += jax.tree_util.tree_reduce(
            lambda x, y: x + y, squared_norms
          ) * config.tr.lambda_mols
        elif mode == "aols":
          pass
        elif mode == "tr":
          # NOTE: currently only supports ks
          if _global:
            normal_vector = jax.grad(ae_loss_fn)(x)[:, :-1]
            loss += jnp.mean(
              jnp.sum(
                normal_vector * (tangent_vector(x) + y_pred[:, :-1]),
                axis=(-2, -1)
              )**2
            ) * lambda_
          else:
            pred = train_state.apply_fn(
              params, inputs.reshape(-1, inputs.shape[-1])
            ).reshape(inputs.shape)
            normal_vector = jax.grad(ae_loss_fn)(inputs)[:, :-1]
            loss += jnp.mean(
              jnp.sum(
                normal_vector * (tangent_vector(inputs) + pred[:, :-1]),
                axis=(-2, -1)
              )**2
            ) * lambda_

        return loss, batch_stats

      if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        if _global:
          (loss, batch_stats), grads = grad_fn(
            train_state.params, train_state.batch_stats, True
          )

          train_state = train_state.apply_gradients(grads=grads)
          train_state = train_state.update_batch_stats(batch_stats)
        else:
          (loss, _), grads = grad_fn(train_state.params, None, True)

          train_state = train_state.apply_gradients(grads=grads)
      else:
        if _global:
          loss, batch_stats = loss_fn(
            train_state.params, train_state.batch_stats, False
          )
        else:
          loss, _ = loss_fn(train_state.params, None, False)

      return loss, train_state

    if mode == "ae" and not _global:
      return
    elif mode == "ae":
      ckpt_path = f"ckpts/{pde}/{dataset}_{mode}_{arch}.pkl"
    else:
      ckpt_path = f"ckpts/{pde}/{dataset}_{config.train.sgs}_{mode}_{arch}.pkl"
    # load_dict = ckpt_path
    # if not os.path.exists(load_dict):
    #   """TODO: no architecture check for the models"""
    #   load_dict = None
    load_dict = None
    train_state, schedule = utils.prepare_unet_train_state(
      config_dict, load_dict, _global
    )
    flat_params = traverse_util.flatten_dict(train_state.params)
    total_params = sum(
      jax.tree_util.tree_map(lambda x: x.size, flat_params).values()
    )
    print(f"total parameters for {mode}_{arch}:", total_params)
    if pde == "ns_channel":
      sim_model = utils.create_ns_channel_simulator(config_dict)
    else:
      _, sim_model = utils.create_fine_coarse_simulator(config_dict)
    if pde != "ns_channel":
      fig_name = f"{pde}_{config.train.sgs}_{mode}_{arch}"
    else:
      fig_name = f"{pde}_{mode}_{arch}"
    augment_inputs_fn = partial(
      utils.augment_inputs, pde=pde, input_labels=input_labels, model=sim_model
    )
    if mode == "tr":
      # add tangent-space regularization
      lambda_ = config.train.lambda_
      ae_train_state, _ = utils.prepare_unet_train_state(
        config_dict, f"ckpts/{pde}/{dataset}_ae_unet.pkl", True, False
      )
      ae_fn = partial(
        ae_train_state.apply_fn_with_bn, {
          "params": ae_train_state.params,
          "batch_stats": ae_train_state.batch_stats
        }
      )

      def ae_loss_fn(x):
        x_pred, _ = ae_fn(x, is_training=False)
        loss = jnp.linalg.norm(x - x_pred, axis=-1)
        # loss = jnp.sum((x - x_pred)**2, axis=-1)
        return jnp.sum(loss)

      @jax.jit
      def tangent_vector(x):
        if pde == "ks":
          return jnp.einsum("ij, ajb -> aib", sim_model.L1, (x[:, :-1]**2)/2) +\
            jnp.einsum("ij, ajb -> aib", 2 * sim_model.L, x[:, :-1])
        elif pde == "ns_hit":
          w_hat = jnp.fft.rfft2(x[..., 0])
          n = x.shape[0]
          w_hat2 = jnp.zeros((n * 2, n + 1), dtype=jnp.complex128)
          psi_hat2 = jnp.zeros((n * 2, n + 1), dtype=jnp.complex128)
          w_hat2 = w_hat2.at[:n // 2, :n // 2 + 1].set(w_hat[:n // 2] * 4)
          w_hat2 = w_hat2.at[-n // 2:, :n // 2 + 1].set(w_hat[n // 2:] * 4)
          psi_hat2 = psi_hat2.at[:n // 2, :n // 2 + 1].set(
            -(w_hat / sim_model.laplacian_)[:n // 2] * 4
          )
          psi_hat2 = psi_hat2.at[-n // 2:, :n // 2 + 1].set(
            -(w_hat / sim_model.laplacian_)[n // 2:] * 4
          )
          wx2 = jnp.fft.irfft2(1j * w_hat2 * sim_model.k2x)
          wy2 = jnp.fft.irfft2(1j * w_hat2 * sim_model.k2y)
          psix2 = jnp.fft.irfft2(1j * psi_hat2 * sim_model.k2x)
          psiy2 = jnp.fft.irfft2(1j * psi_hat2 * sim_model.k2y)
          tmp = jnp.zeros_like(w_hat)
          tmp_ = jnp.fft.rfft2(wx2 * psiy2 - wy2 * psix2)
          tmp = tmp.at[:n // 2].set(tmp_[:n // 2, :n // 2 + 1] / 4)
          tmp = tmp.at[n // 2:].set(tmp_[-n // 2:, :n // 2 + 1] / 4)
          return jnp.fft.irfft2(
            -tmp + sim_model.nu * sim_model.laplacian * w_hat
          )

    # training loop
    iters = tqdm(range(epochs))
    loss_hist = []
    for epoch in iters:
      total_loss = 0
      count = 0
      for batch_inputs, batch_outputs in train_dataloader:
        loss, train_state = train_step(
          jnp.array(batch_inputs), jnp.array(batch_outputs), train_state, True
        )
        total_loss += loss
        count += 1
        if not _global:
          lr = schedule(train_state.step)
          desc_str = f"{lr=:.2e}|{loss=:.4e}"
          iters.set_description_str(desc_str)
          loss_hist.append(loss)
      if _global:
        total_loss /= count
        lr = schedule(train_state.step)
        desc_str = f"{lr=:.2e}|{total_loss=:.4e}"
        iters.set_description_str(desc_str)
        loss_hist.append(total_loss)

      if (epoch + 1) % config.train.save == 0:
        with open(ckpt_path, "wb") as f:
          if _global:
            dict = {
              "params": train_state.params,
              "batch_stats": train_state.batch_stats
            }
          else:
            dict = train_state.params
          pickle.dump(dict, f)

    with open(ckpt_path, "wb") as f:
      if _global:
        dict = {
          "params": train_state.params,
          "batch_stats": train_state.batch_stats
        }
      else:
        dict = train_state.params
      pickle.dump(dict, f)
    val_loss = 0
    for batch_inputs, batch_outputs in test_dataloader:
      loss, train_state = train_step(
        jnp.array(batch_inputs), jnp.array(batch_outputs), train_state, False
      )
      val_loss += loss
    print(f"val loss: {val_loss:.4e}")
    plt.plot(np.linspace(0, epochs, len(loss_hist)), loss_hist)
    plt.yscale("log")
    plt.title(f"loss = {loss_hist[-1]:.3e}")
    plt.savefig(f"results/fig/losshist_{fig_name}.png")
    plt.close()

    # pack the NN for later evaluation
    dim = 2
    inputs_ = inputs
    outputs_ = outputs
    type_ = None
    if pde == "ks":
      dim = 1
      if config.sim.BC == "Dirichlet-Neumann":
        type_ = "pad"
        if _global:
          inputs_ = inputs[:, :-1]
          outputs_ = outputs[:, :-1]
    one_traj_length = inputs.shape[0] // config.sim.case_num
    train_state, schedule = utils.prepare_unet_train_state(
      config_dict, ckpt_path, _global, False
    )
    if _global:

      @jax.jit
      def forward_fn(x):
        y_pred, _ = train_state.apply_fn_with_bn(
          {
            "params": train_state.params,
            "batch_stats": train_state.batch_stats
          },
          x,
          is_training=False
        )
        return y_pred
    else:

      @partial(jax.jit, static_argnums=(1, ))
      def _forward_fn(x, is_aug):
        """forward function for the local model

        The shape of the input and output is aligned with the
        global model for the a-posteriori simulation
        """
        if not is_aug:
          """a-posteriori evaluation"""
          if type_ == "pad":
              if x.ndim == 3:
                  x_ = augment_inputs_fn(x[:, :-1])
                  x_ = jnp.concatenate(
                      [x_, jnp.zeros((x_.shape[0], 1, x_.shape[-1]))], axis=1
                  )
              elif x.ndim == 4:
                  x_ = augment_inputs_fn(x[:, :-1, :])
                  x_ = jnp.concatenate(
                      [x_, jnp.zeros((x_.shape[0], 1, x_.shape[-2], x_.shape[-1]))], axis=1
                  )
          else:
              x_ = augment_inputs_fn(x)
          if x.ndim == 3:
              x_ = x_.reshape(-1, x_.shape[-1])
              return train_state.apply_fn(train_state.params, x_).reshape(x.shape)
          elif x.ndim == 4:
              x_ = x_.reshape(-1, x_.shape[-2], x_.shape[-1])
              return train_state.apply_fn(train_state.params, x_).reshape(*(x.shape[:3]), -1)
        else:
          if x.ndim == 3:
              x_ = x.reshape(-1, x.shape[-1])
              return train_state.apply_fn(train_state.params, x_).reshape(*(x.shape[:2]), -1)
          elif x.ndim == 4:
              x_ = x.reshape(-1, x.shape[-2], x.shape[-1])
              return train_state.apply_fn(train_state.params, x_).reshape(*(x.shape[:3]), -1)

      inputs_ = inputs_[..., 0:1]

    # a-priori evaluation of the regression problem
    if mode == "ae":
      utils.eval_a_priori(
        forward_fn=forward_fn,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        inputs=inputs,
        outputs=outputs,
        dim=dim,
        fig_name=f"reg_{fig_name}",
      )
      return
      
    # a-posteriori evaluation using hybrid simulator
    if not _global:
      forward_fn = partial(_forward_fn, is_aug=True)
    utils.eval_a_priori(
      forward_fn=forward_fn,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      inputs=inputs,
      outputs=outputs,
      dim=dim,
      fig_name=f"reg_{fig_name}",
    )
    if not _global:
      forward_fn = partial(_forward_fn, is_aug=False)
      if inputs_.ndim == 3:
          utils.eval_a_posteriori(
              config_dict=config_dict,
              forward_fn=forward_fn,
              inputs=inputs_[:one_traj_length],
              outputs=outputs_[:one_traj_length],
              dim=dim,
              beta=0.0,
              fig_name=f"sim_{fig_name}",
              _plot=True,
          )
      elif inputs_.ndim == 4:
          utils.eval_a_posteriori(
              config_dict=config_dict,
              forward_fn=forward_fn,
              inputs=inputs_[:one_traj_length, :, :, :],
              outputs=outputs_[:one_traj_length, :, :, :],
              dim=dim,
              beta=0.0,
              fig_name=f"sim_{fig_name}",
              _plot=True,
          )

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-c", "--config", default=None, help="Set the configuration file path."
  )
  args = parser.parse_args()
  with open(f"config/{args.config}.yaml", "r") as file:
    config_dict = yaml.safe_load(file)

  # load dataset
  config = Box(config_dict)
  pde = config.case
  input_labels = config.train.input
  _global = (input_labels == "global")
  epochs = config.train.epochs_global if _global else config.train.epochs_local
  print("start loading data...")
  start = time()
  if _global:
    batch_size = config.train.batch_size_global
    arch = "unet"
  else:
    batch_size = config.train.batch_size_local
    arch = "mlp"
  inputs, outputs, train_dataloader, test_dataloader, dataset = utils.load_data(
    config_dict, batch_size
  )
  print(f"finis loading data with {time() - start:.2f}s...")
  print(f"Problem type: {pde}")
  if pde != "ns_channel":
    print(f"{config.train.sgs}")
  # modes_array = ["ae", "ols", "mols", "aols", "tr"]
  modes_array = ["ols"]

  for _ in modes_array:
    print(f"Training {_}...")
    train(_, epochs)


if __name__ == "__main__":
  main()
