# Copyright 2024 The Flax Authors.
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

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random


class CustomTrainState(TrainState):
  batch_stats: dict

  def apply_fn_with_bn(self, *args, is_training, **nargs):
    output, mutated_vars = self.apply_fn(
      *args,
      **nargs,
      mutable=["batch_stats"],
      rngs={'dropout': jax.random.PRNGKey(2)}
    )
    new_batch_stats = mutated_vars["batch_stats"]
    return output, new_batch_stats

  def update_batch_stats(self, new_batch_stats):
    return self.replace(batch_stats=new_batch_stats)


class MLP(nn.Module):
  output_dim: int
  hidden_dim: int = 32
  dtype: str = jnp.float64

  @nn.compact
  def __call__(self, inputs):
    non_linear = nn.tanh
    x = inputs.reshape(inputs.shape[0], -1)
    x = nn.Dense(self.hidden_dim, param_dtype=self.dtype)(x)
    x = non_linear(x)
    x = nn.Dense(self.hidden_dim, param_dtype=self.dtype)(x)
    x = non_linear(x)
    x = nn.Dense(self.output_dim, param_dtype=self.dtype)(x)
   # y = x + nn.Dense(self.output_dim, param_dtype=self.dtype)(
   #   inputs.reshape(inputs.shape[0], -1)
   # )
   # return y - jnp.mean(y, axis=1, keepdims=True) #reduce 1st moment error
    return x + nn.Dense(self.output_dim, param_dtype=self.dtype)(
      inputs.reshape(inputs.shape[0], -1)
    )



"""cVAE model definitions."""


class vae_Encoder(nn.Module):
  """cVAE Encoder."""

  latents: int
  dtype: str = jnp.float64

  @nn.compact
  def __call__(self, x, c):
    x = jnp.concatenate([x, c], axis=1)
    x = nn.Dense(self.latents, name='fc1', param_dtype=self.dtype)(x)
    x = nn.tanh(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean', param_dtype=self.dtype)(x)
    logvar_x = nn.Dense(
      self.latents, name='fc2_logvar', param_dtype=self.dtype
    )(x)
    return mean_x, logvar_x


class vae_Decoder(nn.Module):
  """cVAE Decoder."""

  latents: int
  features: int
  dtype: str = jnp.float64

  @nn.compact
  def __call__(self, z, c):
    z = jnp.concatenate([z, c], axis=1)
    z = nn.Dense(self.latents, name='fc1', param_dtype=self.dtype)(z)
    z = nn.tanh(z)
    z = nn.Dense(self.features, name='fc2', param_dtype=self.dtype)(z)
    return z


class cVAE(nn.Module):
  """Full cVAE model."""

  latents: int = 128
  features: int = 256
  dtype: str = jnp.float64

  def setup(self):
    self.encoder = vae_Encoder(self.latents, self.dtype)
    self.decoder = vae_Decoder(self.latents, self.features, self.dtype)

  def __call__(self, x, c, z_rng):
    mean, logvar = self.encoder(x, c)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z, c)
    return recon_x, mean, logvar

  def generate(self, z, c):
    return self.decoder(z, c)


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


def model(latents, features):
  return cVAE(latents=latents, features=features)


"""UNet model definitions.
Implementation adapted from https://gitlab.com/1kaiser/jax-unet
"""


class Encoder1D(nn.Module):
  features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, x):
    # Block 1
    z1 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(x)
    z1 = nn.relu(z1)
    z1 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(z1)
    z1 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z1)
    z1 = nn.relu(z1)
    z1_pool = nn.max_pool(z1, window_shape=(2, ), strides=(2, ))

    # Block 2
    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z1_pool)
    z2 = nn.relu(z2)
    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z2)
    z2 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z2)
    z2 = nn.relu(z2)
    z2_pool = nn.max_pool(z2, window_shape=(2, ), strides=(2, ))

    # Block 3
    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z2_pool)
    z3 = nn.relu(z3)
    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z3)
    z3 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z3)
    z3 = nn.relu(z3)
    z3_pool = nn.max_pool(z3, window_shape=(2, ), strides=(2, ))

    # Block 4
    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z3_pool)
    z4 = nn.relu(z4)
    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z4)
    z4 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z4)
    z4 = nn.relu(z4)
    z4_dropout = nn.Dropout(0.5, deterministic=not self.training)(z4)
    z4_pool = nn.max_pool(z4_dropout, window_shape=(2, ), strides=(2, ))

    # Block 5 (bottleneck)
    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z4_pool)
    z5 = nn.relu(z5)
    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z5)
    z5 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z5)
    z5 = nn.relu(z5)
    z5_dropout = nn.Dropout(0.5, deterministic=not self.training)(z5)

    return z1, z2, z3, z4_dropout, z5_dropout


class Decoder1D(nn.Module):
  features: int = 2
  output_features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, z1, z2, z3, z4, z5):
    # Up Block 1
    z6_up = jax.image.resize(
      z5, shape=(z5.shape[0], z5.shape[1] * 2, z5.shape[2]), method='nearest'
    )
    z6 = nn.Conv(self.features * 8, kernel_size=(2, ),
                 param_dtype=self.dtype)(z6_up)
    z6 = nn.relu(z6)
    z6 = jnp.concatenate([z4, z6], axis=-1)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)

    # Up Block 2
    z7_up = jax.image.resize(
      z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2]), method='nearest'
    )
    z7 = nn.Conv(self.features * 4, kernel_size=(2, ),
                 param_dtype=self.dtype)(z7_up)
    z7 = nn.relu(z7)
    z7 = jnp.concatenate([z3, z7], axis=-1)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)

    # Up Block 3
    z8_up = jax.image.resize(
      z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2]), method='nearest'
    )
    z8 = nn.Conv(self.features * 2, kernel_size=(2, ),
                 param_dtype=self.dtype)(z8_up)
    z8 = nn.relu(z8)
    z8 = jnp.concatenate([z2, z8], axis=-1)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, ),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)

    # Up Block 4
    z9_up = jax.image.resize(
      z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2]), method='nearest'
    )
    z9 = nn.Conv(self.features, kernel_size=(2, ),
                 param_dtype=self.dtype)(z9_up)
    z9 = nn.relu(z9)
    z9 = jnp.concatenate([z1, z9], axis=-1)
    z9 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)
    z9 = nn.Conv(
      self.features, kernel_size=(self.kernel_size, ), param_dtype=self.dtype
    )(z9)
    z9 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)

    # Final output
    y = nn.Conv(
      self.output_features, kernel_size=(1, ), param_dtype=self.dtype
    )(z9)
    return y


class Encoder2D(nn.Module):
  features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, x):
    z1 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(x)
    z1 = nn.relu(z1)
    z1 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z1)
    z1 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z1)
    z1 = nn.relu(z1)
    z1_pool = nn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z1_pool)
    z2 = nn.relu(z2)
    z2 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z2)
    z2 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z2)
    z2 = nn.relu(z2)
    z2_pool = nn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z2_pool)
    z3 = nn.relu(z3)
    z3 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z3)
    z3 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z3)
    z3 = nn.relu(z3)
    z3_pool = nn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z3_pool)
    z4 = nn.relu(z4)
    z4 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z4)
    z4 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z4)
    z4 = nn.relu(z4)
    z4_dropout = nn.Dropout(0.5, deterministic=False)(z4)
    z4_pool = nn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z4_pool)
    z5 = nn.relu(z5)
    z5 = nn.Conv(
      self.features * 16,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z5)
    z5 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z5)
    z5 = nn.relu(z5)
    z5_dropout = nn.Dropout(0.5, deterministic=False)(z5)

    return z1, z2, z3, z4_dropout, z5_dropout


class Decoder2D(nn.Module):
  features: int = 2
  output_features: int = 2
  kernel_size: int = 3
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, z1, z2, z3, z4_dropout, z5_dropout):
    z6_up = jax.image.resize(
      z5_dropout,
      shape=(
        z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2,
        z5_dropout.shape[3]
      ),
      method='nearest'
    )
    z6 = nn.Conv(self.features * 8, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z6_up)
    z6 = nn.relu(z6)
    z6 = jnp.concatenate([z4_dropout, z6], axis=3)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)
    z6 = nn.Conv(
      self.features * 8,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z6)
    z6 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z6)
    z6 = nn.relu(z6)

    z7_up = jax.image.resize(
      z6,
      shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
      method='nearest'
    )
    z7 = nn.Conv(self.features * 4, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z7_up)
    z7 = nn.relu(z7)
    z7 = jnp.concatenate([z3, z7], axis=3)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)
    z7 = nn.Conv(
      self.features * 4,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z7)
    z7 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z7)
    z7 = nn.relu(z7)

    z8_up = jax.image.resize(
      z7,
      shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
      method='nearest'
    )
    z8 = nn.Conv(self.features * 2, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z8_up)
    z8 = nn.relu(z8)
    z8 = jnp.concatenate([z2, z8], axis=3)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)
    z8 = nn.Conv(
      self.features * 2,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z8)
    z8 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z8)
    z8 = nn.relu(z8)

    z9_up = jax.image.resize(
      z8,
      shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
      method='nearest'
    )
    z9 = nn.Conv(self.features, kernel_size=(2, 2),
                 param_dtype=self.dtype)(z9_up)
    z9 = nn.relu(z9)
    z9 = jnp.concatenate([z1, z9], axis=3)
    z9 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)
    z9 = nn.Conv(
      self.features,
      kernel_size=(self.kernel_size, self.kernel_size),
      param_dtype=self.dtype
    )(z9)
    z9 = nn.BatchNorm(
      use_running_average=not self.training, param_dtype=self.dtype
    )(z9)
    z9 = nn.relu(z9)

    y = nn.Conv(
      self.output_features, kernel_size=(1, 1), param_dtype=self.dtype
    )(z9)

    return y


class UNet(nn.Module):
  input_features: int = 2
  output_features: int = 2
  DIM: int = 2
  kernel_size: int = 3  # New parameter for kernel size
  dtype: str = jnp.float64
  training: bool = True

  @nn.compact
  def __call__(self, x):
    if self.DIM == 2:
      z1, z2, z3, z4_dropout, z5_dropout = Encoder2D(
        self.input_features * 4, self.kernel_size, self.dtype, self.training
      )(x)
      y = Decoder2D(
        self.input_features * 4, self.output_features, self.kernel_size,
        self.dtype, self.training
      )(z1, z2, z3, z4_dropout, z5_dropout)
    elif self.DIM == 1:
      z1, z2, z3, z4, z5 = Encoder1D(
        self.input_features * 8,
        kernel_size=self.kernel_size,
        dtype=self.dtype,
        training=self.training
      )(x)
      y = Decoder1D(
        self.input_features * 8,
        self.output_features,
        kernel_size=self.kernel_size,
        dtype=self.dtype,
        training=self.training
      )(z1, z2, z3, z4, z5)
      # y = nn.softplus(y)

    return y - jnp.mean(y, axis=1, keepdims=True)
