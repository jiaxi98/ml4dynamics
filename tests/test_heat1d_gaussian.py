import pytest
import jax.numpy as jnp
import jax
from ml4dynamics.dynamics import Heat1D  # Adjust import if needed
jax.config.update("jax_enable_x64", True)

def gaussian_ic(x, x0, sigma, A=1.0):
    return A * jnp.exp(-0.5 * ((x - x0) / sigma) ** 2)

def heat_exact_periodic(u0, t, gamma, L):
    """Exact periodic solution via FFT: u_hat(t)=u_hat(0)*exp(-gamma*k^2*t)"""
    N = u0.shape[0]
    dx = L / N
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
    U0 = jnp.fft.fft(u0)
    Ut = U0 * jnp.exp(-gamma * (k**2) * t)
    return jnp.real(jnp.fft.ifft(Ut))

def evolve_cn(model: Heat1D, u0, nsteps):
    x = u0
    for _ in range(nsteps):
        x = model.CN(x)
    return x

@pytest.mark.parametrize("T", [0.5])
def test_heat1d_cn_matches_exact(T):
    L = 10.0
    N = 256
    dt = 1e-3
    gamma = 0.05

    model = Heat1D(L=L, N=N, T=T, dt=dt, gamma=gamma, rng=jax.random.PRNGKey(0), BC="periodic")
    x = jnp.linspace(0.0, L, N, endpoint=False)
    x0 = L / 2.0
    sigma = L / 20.0
    u0 = gaussian_ic(x, x0, sigma)

    u_exact = heat_exact_periodic(u0, T, gamma, L)
    nsteps = int(jnp.rint(T / dt))
    assert jnp.isclose(T, nsteps * dt, atol=1e-12), "T must be a multiple of dt for this test."
    u_cn = evolve_cn(model, u0, nsteps)

    # 1) Accuracy
    rel_l2 = jnp.linalg.norm(u_cn - u_exact) / jnp.linalg.norm(u_exact)
    assert rel_l2 < 2e-3, f"Relative L2 error too large: {rel_l2}"

    # 2) Mean conservation
    mean0 = jnp.mean(u0)
    meanT = jnp.mean(u_cn)
    assert jnp.abs(meanT - mean0) < 1e-10, f"Mean not conserved: {meanT - mean0}"

    # 3) L2 norm monotonic decay
    checkpoints = [0.0, 0.1*T, 0.2*T, 0.5*T, T]
    norms = []
    u = u0
    steps_per_ckpt = [int(jnp.rint(t / dt)) for t in checkpoints]
    last_s = 0
    for s in steps_per_ckpt:
        for _ in range(s - last_s):
            u = model.CN(u)
        norms.append(float(jnp.linalg.norm(u)))
        last_s = s
    assert all(norms[i] >= norms[i+1] - 1e-12 for i in range(len(norms)-1)), f"L2 norm increased: {norms}"
