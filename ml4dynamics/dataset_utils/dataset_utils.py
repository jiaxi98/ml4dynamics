import jax
import jax.numpy as jnp
import ml_collections
from box import Box

def calc_correction(rd_fine, rd_coarse, nx: float, r: int, uv: jnp.ndarray):
  """
  Args:
    uv: shape = [nx, nx, 2]
  """
  next_step_fine = rd_fine.adi(uv)
  tmp = jnp.zeros((nx // r, nx // r, 2))
  for k in range(r):
    for j in range(r):
      tmp += uv[k::r, j::r]
  tmp = tmp / (r**2)
  next_step_coarse = rd_coarse.adi(tmp)
  next_step_coarse_interp = jnp.concatenate(
    [
      jnp.kron(next_step_coarse[..., 0], jnp.ones((r, r)))[..., None],
      jnp.kron(next_step_coarse[..., 1], jnp.ones((r, r)))[..., None],
    ],
    axis=2
  )

  return next_step_fine - next_step_coarse_interp


def _create_box_filter(N1, N2, r, BC):
  """Create box filter (averaging) operator"""
  res_op = jnp.zeros((N2, N1))
  
  if r == 2:
    raise Exception("Deprecated...")
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r].set(1)
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 2].set(1)
  elif r == 4:
    # stencil = 7
    for i in range(N2):
      res_op = res_op.at[i, i * r:i * r + 7].set(1)
    if BC == "periodic":
      res_op = res_op.at[-1, :3].set(1)
    res_op /= 7 / 4
  elif r == 8:
    # stencil = 12
    for i in range(N2):
      res_op = res_op.at[i, i * r + 3:i * r + 12].set(1)
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 7].set(0)
  
  return res_op


def _create_gaussian_filter(N1, N2, r, BC):
  """Create Gaussian filter operator using same edge logic as box filter"""
  res_op = jnp.zeros((N2, N1))
  # Use same stencil size and logic as box filter, but with Gaussian weights
  if r == 2:
    raise Exception("Deprecated...")

  elif r == 4:

    sigma = r // 2 
    stencil_size = (6 * sigma) + 1
    half_width = stencil_size // 2

    for i in range(N2):
      center = i * r
      start = center - half_width
      end = center + half_width + 1

      # Ensure indices are within bounds
      idxs = jnp.arange(start, end)
      valid = (idxs >= 0) & (idxs < N1)
      idxs = idxs[valid]

      # Calculate Gaussian weights
      distances = idxs - center
      weights = jnp.exp(-0.5 * (distances / sigma) ** 2)
      weights = weights / (jnp.sum(weights) + 1e-12)  # Avoid division by zero

      for k, j in enumerate(idxs):
        res_op = res_op.at[i, j].set(weights[k])

    if BC == "periodic":
      # Handle periodic wraparound for all rows that need it
      for i in range(N2):
        center = i * r
        start = center - half_width
        end = center + half_width + 1
        
        # Check if we need wraparound (indices go outside [0, N1))
        if start < 0 or end > N1:
          # Collect all indices including wraparound
          all_idxs = []
          all_weights = []
          
          for idx in range(start, end):
            if 0 <= idx < N1:
              # Normal index
              all_idxs.append(idx)
            elif idx < 0:
              # Wrap from left
              wrapped_idx = idx + N1
              all_idxs.append(wrapped_idx)
            elif idx >= N1:
              # Wrap from right
              wrapped_idx = idx - N1
              all_idxs.append(wrapped_idx)
            
            # Calculate weight using original distance
            distance = idx - center
            all_weights.append(jnp.exp(-0.5 * (distance / sigma) ** 2))
          
          if len(all_weights) > 0:
            all_weights = jnp.array(all_weights)
            all_weights = all_weights / (jnp.sum(all_weights) + 1e-12)
            
            # Clear the row first
            res_op = res_op.at[i, :].set(0.0)
            
            # Set the wrapped weights
            for k, j in enumerate(all_idxs):
              res_op = res_op.at[i, j].set(all_weights[k])
              
  elif r == 8:
    sigma = r // 2  # sigma = 4
    stencil_size = 21  # larger stencil (2*r+5)
    half_width = stencil_size // 2
    
    for i in range(N2):
      center = i * r + 3  # offset by 3 like original
      start = center - half_width
      end = center + half_width + 1

      # Ensure indices are within bounds
      idxs = jnp.arange(start, end)
      valid = (idxs >= 0) & (idxs < N1)
      idxs = idxs[valid]

      # Calculate Gaussian weights
      distances = idxs - center
      weights = jnp.exp(-0.5 * (distances / sigma) ** 2)
      weights = weights / (jnp.sum(weights) + 1e-12)  # Avoid division by zero

      for k, j in enumerate(idxs):
        res_op = res_op.at[i, j].set(weights[k])

  return res_op


def _create_spectral_filter(N1, N2, r, BC):
  """Create spectral cutoff filter operator"""
  if BC == "Dirichlet-Neumann":
    # For non-periodic BC, use DCT-based spectral filter
    # This is more complex - for now, fall back to a smooth approximation
    res_op = _create_smooth_spectral_filter_nonperiodic(N1, N2, r)
  else:
    # For periodic BC, use FFT-based spectral filter
    res_op = _create_spectral_filter_periodic(N1, N2, r)
  
  return res_op


def _create_spectral_filter_periodic(N1, N2, r):
  """FFT-based spectral filter for periodic BC"""
  # Create filter in frequency domain then transform to spatial
  cutoff_freq = N2 // 2  # Cutoff frequency for coarse grid (Nyquist)
  
  res_op = jnp.zeros((N2, N1))
  for i in range(N2):
    # Create impulse at coarse grid location
    impulse = jnp.zeros(N1)
    impulse = impulse.at[i * r].set(1.0)
    
    # Apply spectral filter in frequency domain
    impulse_hat = jnp.fft.fft(impulse)
    # Low-pass filter: keep low frequencies, zero high frequencies
    filtered_hat = jnp.where(
      jnp.abs(jnp.fft.fftfreq(N1) * N1) <= cutoff_freq,
      impulse_hat, 0.0
    )
    filtered = jnp.real(jnp.fft.ifft(filtered_hat))
    res_op = res_op.at[i, :].set(filtered)
  
  # Normalize
  row_sums = jnp.sum(res_op, axis=1, keepdims=True)
  res_op = res_op / row_sums
  return res_op


def _create_smooth_spectral_filter_nonperiodic(N1, N2, r):
  """Smooth spectral-like filter for non-periodic BC"""
  res_op = jnp.zeros((N2, N1))
  stencil_size = 2 * r + 1
  for i in range(N2):
    center = i * r + r // 2
    start = center - r
    end = center + r + 1
    idxs = jnp.arange(start, end)
    valid = (idxs >= 0) & (idxs < N1)
    idxs = idxs[valid]
    distances = idxs - center
    weights = 0.5 * (1 + jnp.cos(jnp.pi * distances / r))
    weights = weights / (jnp.sum(weights))
    for k, j in enumerate(idxs):
      res_op = res_op.at[i, j].set(weights[k])
  return res_op


def res_int_fn(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  r = config.sim.rx
  if config.case == "ks":
    BC = config.sim.BC
    filter_type = config.sim.get('filter_type', 'box')  # default to box filter
    if BC == "periodic":
      N1 = config.sim.n
    elif BC == "Dirichlet-Neumann":
      N1 = config.sim.n - 1
    N2 = N1 // r
    
    if filter_type == "box":
      # Original box filter implementation
      res_op = _create_box_filter(N1, N2, r, BC)
      res_op /= r
    elif filter_type == "gaussian":
      res_op = _create_gaussian_filter(N1, N2, r, BC)
    elif filter_type == "spectral":
      res_op = _create_spectral_filter(N1, N2, r, BC)
    else:
      raise ValueError(f"Unknown filter_type: {filter_type}")
    
    int_op = jnp.zeros((N1, N2))
    int_op = jnp.linalg.pinv(res_op)
    print('res_op: ', res_op)
    print('int_op: ', int_op)
    assert jnp.allclose(res_op @ int_op, jnp.eye(N2))  #atol=1e-3
    assert jnp.allclose(res_op.sum(axis=-1), jnp.ones(N2)) #atol=1e-6

    @jax.jit
    def res_fn(x):
      return (x.reshape(-1, N1) @ res_op.T).reshape(N2, -1)

    @jax.jit
    def int_fn(x):
      return (x.reshape(-1, N2) @ int_op.T).reshape(N1, -1)
  elif config.case == "react_diff" or config.case == "ns_hit":
    n = config.sim.n

    @jax.jit
    def res_fn(x):
      result = jnp.zeros((n // r, n // r, x.shape[-1]))
      for k in range(r):
        for j in range(r):
          result += x[k::r, j::r]
      return result / (r**2)

    @jax.jit
    def int_fn(x):
      # only works for the case with 2 components
      return jnp.concatenate(
        [
          jnp.kron(x[..., 0], jnp.ones((r, r)))[..., None],
          jnp.kron(x[..., 1], jnp.ones((r, r)))[..., None],
        ],
        axis=2
      )

  return res_fn, int_fn
