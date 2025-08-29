import copy
import os
import time
from abc import ABCMeta, abstractmethod

os.environ['JAX_XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.numpy.linalg import solve
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

# from scipy.linalg import lstsq
# from scipy.sparse import csr_matrix
# from scipy.sparse.linalg import lsqr
"""
Other candidates for chaotic dynamics:
Logistic mapping
van der Pol oscillator, it seems that
double pendulum
Aizawa Attractor
Newton-Leipnik system
Nose-Hoover oscillator
Halvorsen Attractor
Rabinovich-Fabrikant system
Chen-Lee system
3-cell CNN
Bouali system
Finance attractor
Thomas attractor
"""


class dynamics(object):
  """Base class for simulating dynamical systems

  TODO: need to do function programming for this class, i.e.
  write make function for this and package all the params outside
  the class

  NOTE: N denotes the dimension of the state variable of the dynamics and
  will change according to the boundary condition.
  """
  __metaclass__ = ABCMeta

  def __init__(
    self,
    model_type,
    N,
    T=0,
    dt=0.001,
    tol=1e-8,
    init_scale=1e-2,
    tv_scale=1e-8,
    rng=random.PRNGKey(0),
    plot=False
  ):
    super().__init__()
    self.model_type = model_type
    self.N = N  # dimension
    self.T = T
    self.dt = dt
    self.step_num = int(self.T / self.dt)
    self.tol = tol
    self.init_scale = init_scale  # perturbation scale to calculate Lyapunov exp
    self.tv_scale = tv_scale  #
    self.rng = rng
    self.attractor = jnp.zeros(N)
    self.attractor_flag = False
    self.plot = plot
    self.print_ = False

  @abstractmethod
  def f(self, t, x):
    # evolution operator of the dynamics
    # currently we only consider continuous time dynamics \p_t u = f(u)
    # It remains to adjust this framework to discrete time dynamics u^{t+1} = f(u^t)
    # However, it seems that the discrete time dynamics is more suitable to use as the general framework as
    # we always need to solve the equation via discretization
    print("This is an abstract f method!")
    raise NotImplementedError

  @abstractmethod
  def Jacobi(self, x):
    # Jacobian of the evolution operator of the dynamics
    print("This is an abstract Jacobi method!")
    raise NotImplementedError

  @abstractmethod
  def set_attractor(self):
    # Set the value of the attractor
    print("This is an abstract set_attractor method!")
    raise NotImplementedError

  #############################################################################
  # Summary of several iterative solvers:
  # FE & RK4: applicable to compute several trajectories parallelly.
  # ivp: calculate one accurate trajectory once each time.                                                #
  #############################################################################
  def FE(self, t, x):
    # forward Euler scheme
    dt = self.dt
    f = self.f
    x = x + dt * f(t, x)
    return x

  def RK4(self, x):
    # fourth-order Runge-Kutta scheme
    t = 0
    dt = self.dt
    f = self.f
    k1 = f(t, x)
    k2 = f(t, x + dt * k1 / 2)
    k3 = f(t, x + dt * k2 / 2)
    k4 = f(t, x + dt * k3)
    x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

  def ivp(self, method='RK45'):
    self.sol = solve_ivp(
      self.f,
      t_span=[0, self.T],
      y0=self.x,
      method=method,
      t_eval=jnp.linspace(0, self.T, self.step_num + 1),
      vectorized=True
    )

  def run_target_simulation(self, x):
    # NOTE: self.x_targethist should be something like a static member
    # which can only be modified in some static method and is
    # forbidden to be changed in other methods
    # This method is still maintained for inverse problem application, where we first simulate the
    # model using certain parameters and set this as the target trajectory and use optimization
    # method to find the optimal parameters or initial condition.

    step_num = self.step_num
    N = self.N
    iter = self.CN
    self.x_targethist = jnp.zeros([step_num, N])
    self.x_hist = jnp.zeros([step_num, N])
    self.y_hist = jnp.zeros([step_num, N])
    self.w_hist = jnp.zeros([step_num, N])
    for i in range(step_num):
      self.x_targethist = self.x_targethist.at[i].set(x)
      x = iter(x)
    self.x_hist = self.x_hist.at[0].set(self.x_targethist[0])

  def run_simulation(self, x, iter: callable):
    step_num = self.step_num
    self.x_hist = np.zeros([step_num, *x.shape])
    for i in range(step_num):
      self.x_hist[i] = x
      x = iter(x)

    # self.check_simulation()

  def run_simulation_with_correction(self, x, iter, corrector):
    step_num = self.step_num
    self.x_hist = jnp.zeros([step_num, self.N])
    for i in range(step_num):
      self.x_hist = self.x_hist.at[i].set(x)
      x = iter(x) + corrector(x) * self.dt

    # postprocess for visualization
    self.x_hist = jnp.where(self.x_hist < 20, self.x_hist, 20)
    self.x_hist = jnp.where(self.x_hist > -20, self.x_hist, -20)
    # self.check_simulation()

  def run_simulation_with_probabilistic_correction(self, x, iter, corrector):
    step_num = self.step_num
    self.x_hist = jnp.zeros([step_num, self.N])
    rng = self.rng
    for i in range(step_num):
      rng, key = random.split(rng)
      self.x_hist = self.x_hist.at[i].set(x)
      x = iter(x) + corrector(x, key) * self.dt

    # postprocess for visualization
    self.x_hist = jnp.where(self.x_hist < 20, self.x_hist, 20)
    self.x_hist = jnp.where(self.x_hist > -20, self.x_hist, -20)
    # self.check_simulation()

  def check_simulation(self):
    if jnp.any(jnp.isnan(self.x_hist)) or jnp.any(jnp.isinf(self.x_hist)):
      raise Exception("The simulation contains Inf or NaN")

  def delay_embedding(self, observed_dim=0, method='RK45'):
    # observed_dim is the observed dimension, default to be 'x' coordinate
    # but currently it seems that the visuallization is hard as the delayed dimension is too high
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
      self.sol.y[observed_dim, :-2],
      self.sol.y[observed_dim, 1:-1],
      self.sol.y[observed_dim, 2:],
      c=jnp.linspace(1, 5, self.T),
      s=jnp.linspace(1, 5, self.T)
    )
    cbar = plt.colorbar(sc)
    plt.show()

  def lyapunov_exp(self):
    # Lyapunov calculator
    dt = self.dt
    T = self.T
    f = self.f
    iter = self.CN
    if ~self.attractor_flag:
      self.set_attractor()
    x = self.attractor + self.init_scale * random.normal(
      self.rng, shape=(self.N, )
    )
    x_ = x + self.tv_scale * random.normal(self.rng, shape=(self.N, ))
    l_exp = 0

    for t in jnp.arange(0, T, dt):
      d0 = jnp.linalg.norm(x - x_)
      x = iter(x)
      x_ = iter(x_)
      d1 = jnp.linalg.norm(x - x_)
      l_exp = l_exp + jnp.log2(d1 / d0)
      x_ = x + (x_ - x) * d0 / d1
      #self.x_hist[:, int(t/100)] = x
      if int(t / dt + 1) % int(T // 500 / dt) == 0 and self.plot:
        self.x_hist[:, int((t / dt + 1) // (T // 500 / dt)) - 1] = x
        if int(t / dt + 1) % int(T // 50 / dt) == 0:
          print(
            "The current numerical value of Lyapunov exp is {:.4f}".format(
              l_exp / t / dt
            )
          )

  def calc_attractor_dimension(self):
    epsilon = jnp.logspace(-2, -1, 10)
    count = jnp.zeros(epsilon.shape)
    for i in range(epsilon.shape[0]):
      count[i], spanning_set = epsilon_spanning_set(self.sol.y.T, epsilon[i])
    jnp.savez(
      '../result/dim.jnpz', T=self.T, dt=self.dt, epsilon=epsilon, count=count
    )

  def calc_attractor_dimension2(self, epsilon=0.1):
    # in order to use this method, the data has to be very dense, i.e.
    # we need use very small dt to simulate the dynamics for a relative long time T
    kdtree = cKDTree(self.sol.y.T)
    sample_num = 100
    dim = []

    for i in range(sample_num):
      point = self.sol.y.T[i]
      nodes_num1 = len(kdtree.query_ball_point(point, epsilon))
      nodes_num2 = len(kdtree.query_ball_point(point, epsilon * 2))
      dim.append(jnp.log2(nodes_num2 / nodes_num1))
    jnp.savez(
      '../result/dim2.jnpz', T=self.T, dt=self.dt, epsilon=epsilon, dim=dim
    )
    return dim

  def delay_embedding(self, x):
    '''
        delay embedding of the dynamics via Takens' embedding theorem
        a good reference: https://towardsdatascience.com/time-series-forecasting-with-dynamical-systems-methods-fa4afdf16fd0 which discuss some method to choose the delay time and embedding dimension
        '''
    raise NotImplementedError

  def sensitivity_analysis(self, dfds, discretization='CN'):
    N = self.N
    n = self.step_num
    dt = self.dt
    J = self.Jacobi
    R = jnp.eye(n * N)
    b = jnp.zeros((n - 1) * N)
    sol = jnp.zeros([N * n])
    for i in range(n - 1):
      if discretization == 'FE':
        # forward Euler discretization
        sol[(i + 1) * N:(i + 2) *
            N] = (jnp.eye(N) + dt * J(self.x_hist[:, i])
                  ) @ sol[i * N:(i + 1) * N] + dfds(self.x_hist[:, i])
      elif discretization == 'BE':
        # backward Euler discretization
        sol[(i + 1) * N:(i + 2) * N] = solve(
          jnp.eye(N) - dt * J(self.x_hist[:, i + 1]),
          sol[i * N:(i + 1) * N] + dfds(self.x_hist[:, i + 1])
        )
      elif discretization == 'CN':
        # CN discretization
        sol[(i + 1) * N:(i + 2) * N] = solve(
          jnp.eye(N) - dt / 2 * J(self.x_hist[:, i + 1]),
          (jnp.eye(N) + dt / 2 * J(self.x_hist[:, i])) @ sol[i * N:(i + 1) * N]
          + dfds(self.x_hist[:, i + 1]) / 2 + dfds(self.x_hist[:, i]) / 2
        )

    return sol.reshape(1, N * n)

  def lss(self, dt: float, step_num: int, discretization='CN'):
    """LSS method to calculate the tangent direction:
        Before calling this function, make sure that the forward equation is simulated to obtain a trajectory. 
        Currently, the size is NT \times NT which is very large, but we can do downsampling 
        on the temporal direction to reduce the system size. Here is the discretization of linearized equation

        Args:
            dt: time step size of the LSS discretization.
            step_num: number of time steps of the LSS discretization.
        Returns:
            sol: the solution object of the least square system, the shadowing direction array
                is sol[0].reshape(N, N, step_num).

        $$
        \begin{pmatrix}
          \mfI & -\nabla_u f(u_{T-1}) & 0 & \cdots & 0 & 0 \\
          0 & \mfI & -\nabla_u f(u_{T-2}) & \cdots & 0 & 0 \\
          0 & 0 & \mfI & \cdots & 0 & 0 \\
          \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
          0 & 0 & 0 & \cdots & \mfI & -\nabla_u f(u_1)	\\
          0 & 0 & 0 & \cdots & 0 & 0
        \end{pmatrix}\begin{pmatrix}
          v_T \\ v_{T-1} \\ v_{T-2} \\ \vdots \\ v_2 \\ v_1
        \end{pmatrix} = \begin{pmatrix}
          \p_s f(u_{T-1}) \\ \p_s f(u_{T-2}) \\ \p_s f(u_{T-3}) \\ \vdots \\ \p_s f(u_{1}) \\ 0
        \end{pmatrix}.
        $$

    """

    # the notation here is not consistent, for 1D problem, the size is NT while for
    # 2D problem the size should be N^2
    # TODO: maybe we should change the model_type to DIM
    if self.model_type == 'Lorenz' or self.model_type == 'Rossler' or self.model_type == 'KS':
      N = self.N
    elif self.model_type == 'NS':
      N = self.N**2
    n = int(step_num)
    r = self.step_num // n
    J = self.Jacobi
    print('size of the least square system: {}'.format(n * N))
    R = jnp.eye(n * N)
    b = jnp.zeros((n - 1) * N)
    # here we move the external defined dfds function to a class method, currently only implemented for
    # NS equation and Lorenz equation, would be better to implement also for other dynamics
    dfds = self.dfds
    T1 = time.perf_counter()
    for i in range(n - 1):
      if discretization == 'FE':
        # forward Euler discretization
        R[i * N:(i + 1) * N, (i + 1) * N:(i + 2) *
          N] = -jnp.eye(N).to(self.device
                              ) - dt * J(self.x_hist[..., (n - 2 - i) * r])
        b[i * N:(i + 1) * N] = dfds(self.x_hist[..., (n - 2 - i) * r]) * dt
      elif discretization == 'BE':
        # backward Euler discretization
        R[i * N:(i + 1) * N, (i + 1) * N:(i + 2) * N] = -jnp.eye(N)
        R[i * N:(i + 1) * N,
          i * N:(i + 1) * N] = R[i * N:(i + 1) * N, i * N:(i + 1) *
                                 N] - dt * J(self.x_hist[..., (n - 1 - i) * r])
        b[i * N:(i + 1) * N] = dfds(self.x_hist[..., (n - 1 - i) * r]) * dt
      elif discretization == 'CN':
        # CN discretization
        R[i * N:(i + 1) * N, (i + 1) * N:(i + 2) *
          N] = -jnp.eye(N) - dt * J(self.x_hist[..., (n - 2 - i) * r]) / 2
        R[i * N:(i + 1) * N, i * N:(i + 1) *
          N] = R[i * N:(i + 1) * N, i * N:(i + 1) *
                 N] - dt * J(self.x_hist[..., (n - 1 - i) * r]) / 2
        b[i * N:(i + 1) * N] = (
          dfds(self.x_hist[..., (n - 1 - i) * r]) +
          dfds(self.x_hist[..., (n - 2 - i) * r])
        ) / 2 * dt
    T2 = time.perf_counter()
    print('Assembly time: {:4e}'.format(T2 - T1))
    #print('Condition number: {:.2e}'.format(jnp.linalg.cond(R[:(n-1)*N])))
    #T3 = time.perf_counter()
    #print('Conditioning time: {:4e}'.format(T3 - T2))
    #print('Norm of the residue vector: {:.2e}'.format(jnp.linalg.norm(b)))
    #tmp = (self.x_hist - self.x_targethist)[:,::r].reshape(n*N)
    #print('Rayleigh quotient of the residue vector: {:.2e}'.format(tmp.T @ R @ tmp/(tmp.T @ tmp)))
    #sol = lsqr(csr_matrix(R[:(n-1)*N]), b)
    sol = lstsq(R[:(n - 1) * N], b)
    T3 = time.perf_counter()
    print('Solving time: {:4e}'.format(T3 - T2))
    return sol


class Rossler(dynamics):
  '''
    https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
    '''

  def __init__(
    self,
    model_type='Rossler',
    N=3,
    T=100,
    dt=0.001,
    a=0.1,
    b=0.1,
    c=14,
    tol=1e-8,
    init_scale=1,
    tv_scale=1e-8,
    plot=False
  ):
    super(Rossler,
          self).__init__(model_type, N, T, dt, tol, init_scale, tv_scale, plot)
    self.a = a
    self.b = b
    self.c = c

  def f(self, t, x):
    # parallel version
    fx = jnp.zeros(x.shape)
    fx[0] = -x[1] - x[2]
    fx[1] = x[0] + self.a * x[1]
    fx[2] = self.b + x[2] * (x[0] - self.c)
    return fx

  def Jacobi(self, x):
    jacobi = jnp.zeros([self.N, self.N])
    jacobi[0, 1] = -1
    jacobi[0, 2] = -1
    jacobi[1, 0] = 1
    jacobi[1, 1] = self.a
    jacobi[2, 0] = x[2]
    jacobi[2, 2] = x[0] - self.c
    return jacobi

  def set_attractor(self):
    a = self.a
    b = self.b
    c = self.c
    if c**2 < 4 * a * b:
      print('No fixed point!!!')
    else:
      self.attractor = jnp.zeros(self.N)
      self.attractor[0] = (c + jnp.sqrt(c**2 - 4 * a * b)) / 2
      self.attractor[1] = -(c + jnp.sqrt(c**2 - 4 * a * b)) / 2 / a
      self.attractor[2] = (c + jnp.sqrt(c**2 - 4 * a * b)) / 2 / a
    self.attractor_flag = True


class Lorenz(dynamics):

  def __init__(
    self,
    model_type='Lorenz',
    N=3,
    T=100,
    dt=0.001,
    sigma=10,
    rho=28,
    beta=8 / 3,
    tol=1e-8,
    init_scale=1,
    tv_scale=1e-8,
    plot=False
  ):
    super(Lorenz,
          self).__init__(model_type, N, T, dt, tol, init_scale, tv_scale, plot)
    self.sigma = sigma
    self.rho = rho
    self.beta = beta

  def f(self, t, x):
    # parallel version
    fx = jnp.zeros(x.shape)
    fx[0] = self.sigma * (x[1] - x[0])
    fx[1] = x[0] * (self.rho - x[2]) - x[1]
    fx[2] = x[0] * x[1] - self.beta * x[2]
    return fx

  def Jacobi(self, x):
    jacobi = jnp.zeros([self.N, self.N])
    jacobi[0, 0] = -self.sigma
    jacobi[0, 1] = self.sigma
    jacobi[1, 0] = self.rho - x[2]
    jacobi[1, 1] = -1
    jacobi[1, 2] = -x[0]
    jacobi[2, 0] = x[1]
    jacobi[2, 1] = x[0]
    jacobi[2, 2] = -self.beta
    return jacobi

  def set_attractor(self):
    """
        The attractor is set for the purpose to calculate the Lyapunov exponent and 
        the intrinsic dimension of the dynamics
        """
    self.attractor = jnp.zeros(self.N)
    if self.rho > 1:
      self.attractor[0] = -jnp.sqrt(self.beta * (self.rho - 1))
      self.attractor[1] = -jnp.sqrt(self.beta * (self.rho - 1))
      self.attractor[2] = self.rho - 1
    self.attractor_flag = True

  def dfds(self, x):
    # ijnput: x is the 3D state vector of the Lorenz model,
    # output: dfd\rho where \rho is the Rayleigh number of the Lorenz model
    output = jnp.zeros(x.shape)
    output[1] = x[0]
    return output


class KS(dynamics):
  r"""Kuramoto–Sivashinsky equation

  The 1d version of the Kuramoto–Sivashinsky equation is:

  $$
    u_t + (c + u)u_x + u_{xx} + \nu u_{xxxx} = 0.
  $$

  """

  def __init__(
    self,
    model_type='KS',
    L=100 * jnp.pi,
    N=512,
    T=10,
    dt=0.01,
    nu=1,
    c=1,
    BC="periodic",
    tol=1e-8,
    init_scale=1e-2,
    tv_scale=1e-8,
    rng=random.PRNGKey(0),
    plot=False
  ):
    super(KS, self).__init__(
      model_type, N, T, dt, tol, init_scale, tv_scale, rng, plot
    )
    self.Lx = L
    self.nu = nu
    self.c = c
    dt = self.dt
    self.BC = BC
    if self.BC == "periodic":
      self.dx = self.Lx / self.N
    elif self.BC == "Dirichlet-Neumann":
      self.dx = self.Lx / (self.N + 1)
    self.assembly_matrix()

  def f(self, t, x):
    return (jnp.eye(self.N) - 2 * self.L * self.dt) @ x -\
      self.dt * self.L1 @ (x**2)

  def assembly_matrix(self):

    N = self.N
    dx = self.dx
    # periodic BC
    L1 = jnp.roll(jnp.eye(N), 1, axis=1) - jnp.roll(jnp.eye(N), -1, axis=1)
    L2 = jnp.roll(jnp.eye(N), 1, axis=1) + jnp.roll(jnp.eye(N), -1, axis=1) -\
      jnp.eye(N) * 2
    L4 = jnp.roll(jnp.eye(N), 2, axis=1) + jnp.roll(jnp.eye(N), -2, axis=1) -\
      4*jnp.roll(jnp.eye(N), 1, axis=1) - 4*jnp.roll(jnp.eye(N), -1, axis=1) +\
      jnp.eye(N) * 6

    if self.BC == "Dirichlet-Neumann":
      # Dirichlet & Neumann BC following https://arxiv.org/pdf/1307.8197
      L1 = L1.at[0, -1].set(0)
      L1 = L1.at[-1, 0].set(0)
      L2 = L2.at[0, -1].set(0)
      L2 = L2.at[-1, 0].set(0)
      L4 = L4.at[0, -1].set(0)
      L4 = L4.at[0, 0].set(7)
      L4 = L4.at[0, -2].set(0)
      L4 = L4.at[1, -1].set(0)
      L4 = L4.at[-2, 0].set(0)
      L4 = L4.at[-1, -1].set(7)
      L4 = L4.at[-1, 0].set(0)
      L4 = L4.at[-1, 1].set(0)

    self.L1 = L1 / 2 / dx
    self.L2 = L2 / dx**2
    self.L4 = L4 / dx**4
    self.L = self.c / 2 * self.L1 + self.L2 / 2 + self.nu * self.L4 / 2

  def Jacobi(self, x):
    # u_x w term
    jacobi = jnp.diag(self.L1 @ x) +     \
                self.L2 + \
                self.nu * self.L4 + \
                jnp.diag(x+self.c) @ self.L1
    return jacobi

  def set_attractor(self):
    x_hat = jnp.zeros(self.N // 2 + 1)
    x_hat = x_hat.at[5].set(self.N)
    x_hat = x_hat.at[10].set(self.N * 2)
    x = jnp.fft.irfft(x_hat)
    iter = self.CN
    for t in jnp.arange(0, 100, self.dt):
      x = iter(x)
    self.attractor = copy.deepcopy(x)
    self.attractor_flag = True

  # @jax.jit
  # NOTE: currently we can not jit as the self is not an array
  def CN_FEM(self, x):
    return jax.scipy.linalg.solve(
      jnp.eye(self.N) + self.L * self.dt,
      (jnp.eye(self.N) - self.L * self.dt) @ x -\
      self.dt / 2 * self.L1 @ (x**2)
    )

  def CN_FEM_test(self, x):
    """Test the solver of the KS equation using analytic formula
    with Crank-Nicolson scheme
    """
    return jax.scipy.linalg.solve(
      jnp.eye(self.N) + self.L * self.dt,
      (jnp.eye(self.N) - self.L * self.dt) @ x -\
      self.dt / 2 * self.L1 @ (x**2) - self.dt * self.source
    )

  def FE_test(self, x):
    """Test the solver of the KS equation using analytic formula
    with Forward Euler scheme
    """
    return x + self.dt * (
      self.c * self.L1 @ x + self.L2 @ x + self.nu * self.L4 @ x +
      self.L1 @ (x**2) / 2 + self.source
    )

  def CN_FEM_adj(self, x, i):
    delta_x = self.x_hist[:, i] - self.x_targethist[:, i]
    return jax.scipy.linalg.solve(
      self.Lplus, self.Lmimus @ x +
      self.dt * self.L1 @ (x * self.x_hist[:, i]) - self.dt * delta_x
    )

  def CN(self, x):
    # Crank-Nicolson scheme for spectral method with periodic boundary condition,
    # however, periodic boundary condition does not give ergodic system
    dt = self.dt
    c = self.c
    k = jnp.fft.rfftfreq(self.N, d=self.Lx / self.N) * 2 * jnp.pi
    x_hat = jnp.fft.rfft(x)
    x_hat = ((1 - c*1j*k/2 + dt/2 * (k**2) - self.nu*dt/2 * (k**4))*x_hat -\
      dt/2 * 1j * k * jnp.fft.rfft(x*x))\
      /(1 + c*1j*k/2 - dt/2 * (k**2) + self.nu*dt/2 * (k**4))
    x = jnp.fft.irfft(x_hat)
    return x

  def CN_adj(self, lambda_, i):
    # CN scheme for the dual variable \lambda
    dt = self.dt
    c = self.c
    k = jnp.fft.rfftfreq(self.N, d=self.Lx / self.N) * 2 * jnp.pi
    lambda_hat = jnp.fft.rfft(lambda_)
    delta_x = jnp.fft.rfft(self.x_hist[:, i] - self.x_targethist[:, i])
    lambda_hat = ((1 - c*1j*k/2 + dt/2 * (k**2) - self.nu*dt/2 * (k**4))*\
      lambda_hat - dt * 1j * k * jnp.fft.rfft(lambda_*self.x_hist[:, i]) -\
      dt * delta_x)/(1 + c*1j*k/2 - dt/2 * (k**2) + self.nu*dt/2 * (k**4))
    lambda_ = jnp.fft.irfft(lambda_hat)
    return lambda_

  def calc_gradient(self):
    dt = self.dt
    step_num = self.step_num
    iter = self.CN_FEM
    dual_iter = self.CN_FEM_adj
    k = jnp.fft.rfftfreq(self.N, d=self.Lx / self.N) * 2 * jnp.pi
    # saving dual variable instead of primal variable helps save the memory consumption
    x = copy.deepcopy(self.x)
    y = jnp.zeros(self.N)

    # solve the primal equation
    for i in range(step_num):
      self.x_hist[:, i] = copy.deepcopy(x)
      #print(jnp.linalg.norm(self.x_hist[:, i]-self.x_targethist[:, i]))
      x = iter(x)

    # solve the dual equation
    for i in range(step_num):
      self.y_hist[:, step_num - i - 1] = copy.deepcopy(y)
      y = dual_iter(y, step_num - i - 1)

    #loss = jnp.linalg.norm(self.x_hist - self.x_targethist, ord='fro')**2/2/self.N/step_num
    loss = jnp.mean((self.x_hist - self.x_targethist)**2) / 2

    gradient = 0
    bar = 0.5
    # we implement a SGD
    for i in range(step_num):
      #gradient = gradient + jnp.sum(self.y_hist[:,i]*irfft(rfft(self.x_hist[:,i])*(k**4))) * (1 if (r.rand()>bar) else 0)
      gradient = gradient + jnp.sum(
        self.y_hist[:, i] *
        jnp.fft.irfft(jnp.fft.rfft(self.x_hist[:, i]) * (k**4))
      ) * (1 if (random.normal(self.rng) > bar) else 0)
    gradient = gradient / step_num / self.N / (1 - bar)
    if self.print_:
      print("The numerical value of nu is {:.2e}".format(self.nu))
      print("The numerical error of loss function is {:.2e}".format(loss))
      print("The numerical value of the gradient is {:.2e}".format(gradient))

    return gradient, loss

  ########################################################################################################################
  # The following methods are deprecated
  ########################################################################################################################
  def CN_lss(self, w, i):
    # CN scheme for the modified equation of LSS method
    dt = self.dt
    k = jnp.fft.rfftfreq(self.N, d=self.Lx / self.N) * 2 * jnp.pi
    w_hat = jnp.fft.rfft(w)
    # I am not sure whether here should be
    u0 = copy.deepcopy(self.x_hist[:, i])
    u0_hat = jnp.fft.rfft(u0)
    w_hat = ((1 + dt/2 * (k**2) - self.nu*dt/2 * (k**4))*w_hat
             - dt * (k**4) * u0_hat
             - dt * jnp.fft.rfft(u0 * jnp.fft.irfft(1j * k * w_hat))
             - dt * jnp.fft.rfft(w * jnp.fft.irfft(1j * k * u0_hat)))\
        /(1 - dt/2 * (k**2) + self.nu*dt/2 * (k**4))
    w = jnp.fft.irfft(w_hat)
    return w

  def CN_lss_adj(self, w, i):
    # CN scheme for the modified equation of LSS method
    dt = self.dt
    k = jnp.fft.rfftfreq(self.N, d=self.Lx / self.N) * 2 * jnp.pi
    w_hat = jnp.fft.rfft(w)
    # I am not sure whether here should be
    u0 = copy.deepcopy(self.x_hist[:, i])
    u0_hat = jnp.fft.rfft(u0)
    w_hat = ((1 + dt/2 * (k**2) - self.nu*dt/2 * (k**4))*w_hat
             - dt * (k**4) * u0_hat
             - dt * jnp.fft.rfft(u0 * jnp.fft.irfft(1j * k * w_hat))
             - dt * jnp.fft.rfft(w * jnp.fft.irfft(1j * k * u0_hat)))\
        /(1 - dt/2 * (k**2) + self.nu*dt/2 * (k**4))
    w = jnp.fft.irfft(w_hat)
    return w

  def calc_init_condition(self):
    # this function calculates the optimal initial condition
    # given \nu and target trajectories
    dt = self.dt
    T = self.T
    iter = self.CN
    dual_iter = self.CN_adj
    k = jnp.fft.rfftfreq(self.N, d=self.Lx / self.N) * 2 * jnp.pi
    # saving dual variable instead of primal variable helps save the memory consumption
    x = copy.deepcopy(self.x_hist[:, 0])
    y = jnp.zeros(self.N)
    iter_num = 30
    step_size = .01

    for k in range(iter_num):
      # solve the primal equation
      for i in range(int(T / dt)):
        self.x_hist[:, i] = copy.deepcopy(x)
        #print(jnp.linalg.norm(self.x_hist[:, i]-self.x_targethist[:, i]))
        x = iter(x)

      # solve the dual equation
      for i in range(int(T / dt)):
        self.y_hist[:, int(T / dt) - i - 1] = copy.deepcopy(y)
        y = dual_iter(y, int(T / dt) - i - 1)

      self.x_hist[:, 0] = self.x_hist[:, 0] - self.y_hist[:, 0] / (
        i / 10 + 1
      ) / T / self.L * step_size

  def calc_gradient_lss(self):

    dt = self.dt
    T = self.T
    iter = self.CN
    iter_lss = self.CN_lss
    dual_iter = self.CN_lss_adj
    k = jnp.fft.rfftfreq(self.N, d=self.Lx / self.N) * 2 * jnp.pi
    # saving dual variable instead of primal variable helps save the memory consumption
    x = copy.deepcopy(self.x_hist[:, 0])
    y = jnp.zeros(self.N)
    # I am not sure how to move the initial condition?
    self.w_hist_ = jnp.zeros([self.N, int(T / dt)])
    w = copy.deepcopy(self.w_hist[:, 0])
    #w = jnp.zeros(self.N)

    self.calc_init_condition(self)

    self.iter_num = 5
    iter_num = self.iter_num
    # root finding using secant metehod
    for j in range(iter_num):
      # solve the LSS primal equation
      for i in range(int(T / dt)):
        self.w_hist[:, i] = copy.deepcopy(w)
        w = iter_lss(w, i)

      # solve the dual equation
      for i in range(int(T / dt)):
        self.y_hist[:, int(T / dt) - i - 1] = copy.deepcopy(y)
        y = dual_iter(y, int(T / dt) - i - 1)

    loss = jnp.linalg.norm(
      self.x_hist - self.x_targethist, ord='fro'
    )**2 / 2 / self.L / self.T

    gradient = 0
    bar = .5
    # we implement a SGD
    for i in range(int(T / dt)):
      gradient = gradient + jnp.sum(
        self.w_hist[:, i] * (self.x_hist[:, i] - self.x_targethist[:, i])
      ) * (1 if (random.normal(self.rng) > bar) else 0)
      #print(jnp.sum(self.y_hist[:,i]*irfft(rfft(self.x_hist[:,i])*(k**4))))
    gradient = gradient / T / self.L
    if self.print_:
      print("The numerical value of nu is {:.4f}".format(self.nu))
      print("The numerical error of loss function is {:.4f}".format(loss))
      print("The numerical value of the gradient is {:.4f}".format(gradient))

    #return gradient
    return -gradient, loss


class react_diff(dynamics):
  r"""
  
  $$
    \frac{\patial \mathbf{u}}{\partial t} = D \Delta \mfu + 
    \phi(\mathbf{u}), \quad T \in [0, 20], 	\\
		\phi(\mathbf{u}) = \phi(u, v) = \begin{pmatrix}
			u - u^3 - v + \alpha	\\
			\beta(u - v)
    \end{pmatrix}
  $$

  The dimension of the state variable is given by $N = 2n^2$ where $n$ is the
  spatial discretization size.
  """

  def __init__(
    self,
    model_type='react_diff',
    L=2 * jnp.pi,
    N=128**2 * 2,
    T=10,
    dt=0.01,
    alpha=0.01,
    beta=1.0,
    gamma=0.05,
    d=2,
    tol=1e-8,
    init_scale=4,
    tv_scale=1e-8,
    plot=False
  ):
    super(react_diff,
          self).__init__(model_type, N, T, dt, tol, init_scale, tv_scale, plot)

    self.L = L
    self.n = int(jnp.sqrt(self.N / 2))
    self.dx = L / self.n
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.d = d
    self.assembly_matrix()

  def f(self, x):
    print("This is an abstract f method!")
    raise NotImplementedError

  def assembly_matrix(self):
    """assemble matrices used in the calculation
    A1 = I - gamma dt \Delta, used in implicit discretization of diffusion term, size n2*n2
    A2 = I - gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    A3 = I + gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    D, size 4n2*n2, Jacobi of the Newton solver in CN discretization
    :d: ratio between the diffusion coeff for u & v
    """

    n = self.n
    gamma = self.gamma
    dt = self.dt
    d = self.d
    dx = self.dx

    L = jnp.eye(n) * -2 + jnp.eye(n, k=1) + jnp.eye(n, k=-1)
    L = L.at[0, -1].set(1)
    L = L.at[-1, 0].set(1)
    L = L / (dx**2)

    # matrix for ADI scheme
    self.L_uminus = jnp.eye(n) - L * gamma * dt / 2
    self.L_uplus = jnp.eye(n) + L * gamma * dt / 2
    self.L_vminus = jnp.eye(n) - L * gamma * dt / 2 * d
    self.L_vplus = jnp.eye(n) + L * gamma * dt / 2 * d

  def adi(self, u):

    dt = self.dt
    rhsu = self.L_uplus @ u[..., 0] @ self.L_uplus +\
      dt * (u[..., 0] - u[..., 1] - u[..., 0]**3 + self.alpha)
    rhsv = self.L_vplus @ u[..., 1] @ self.L_vplus +\
      self.beta * dt * (u[..., 0] - u[..., 1])

    result = jnp.concatenate(
      [
        (solve(self.L_uminus,
               solve(self.L_uminus, rhsu).T).T)[..., None],
        (solve(self.L_uminus,
               solve(self.L_vminus, rhsv).T).T)[..., None]
      ],
      axis=2
    )
    return result


class ns_hit(dynamics):

  def __init__(
    self,
    model_type='NS',
    L=2 * np.pi,
    N=128,
    T=10,
    dt=0.01,
    nu=0.0001,
    tol=1e-8,
    init_scale=4,
    tv_scale=1e-8,
    plot=False
  ):
    super(ns_hit,
          self).__init__(model_type, N, T, dt, tol, init_scale, tv_scale, plot)

    self.L = L
    self.dx = self.L / self.N
    self.dt = dt
    self.nu = nu
    self.assembly_spectral()

  def assembly_spectral(self):
    # assembly array for spectral method
    N = self.N
    L = self.L
    x = np.linspace(0, L - L / N, N)
    y = np.linspace(0, L - L / N, N)
    self.X, self.Y = np.meshgrid(x, y, indexing='ij')
    self.X = self.X
    self.Y = self.Y
    kx = np.fft.rfftfreq(N, d=L / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    self.kx, self.ky = np.meshgrid(ky, kx, indexing='ij')
    self.kx = self.kx
    self.ky = self.ky
    k2x = np.fft.rfftfreq(N * 2, d=L / N * 2) * 2 * np.pi
    k2y = np.fft.fftfreq(N * 2, d=L / N * 2) * 2 * np.pi
    self.k2x, self.k2y = np.meshgrid(k2y, k2x, indexing='ij')
    self.k2x = self.k2x
    self.k2y = self.k2y
    self.laplacian = -(self.kx**2 + self.ky**2)
    self.laplacian_ = self.laplacian.copy()
    self.laplacian_[0, 0] = 1

  def assembly_matrix(self):
    N = self.N
    L = self.L
    dx = self.dx
    L1 = np.zeros((N, N))
    L2 = np.zeros((N, N))

    # assembly matrix for FDM
    # this is for periodic boundary condition#
    for i in range(N):
      L1[i, (i + 1) % N] = 1
      L1[i, (i - 1) % N] = -1
      L2[i, i] = -2
      L2[i, (i + 1) % N] = 1
      L2[i, (i - 1) % N] = 1

    L1 = L1 / 2 / dx
    self.Lx = np.kron(L1, np.eye(N))
    self.Ly = np.kron(np.eye(N), L1)
    L2 = L2 / dx**2
    self.L2 = np.kron(L2, np.eye(N)) + np.kron(np.eye(N), L2)
    self.L2_inv = np.linalg.pinv(self.L2)
    #self.L2 = self.L2.to_sparse_csr()

  def Jacobi(self, w):
    # This Jacobi matrix is based on flatten the 2D state variable w
    # input: w is the vorticity in physical space of size N \times N,
    # output:
    w_hat = np.fft.rfft2(w)
    psi = -w_hat / self.laplacian_
    u = np.zeros([2, self.N**2], device=self.device)
    u[0] = np.fft.irfft2(1j * psi * self.ky).reshape(self.N**2)
    u[1] = np.fft.irfft2(1j * psi * self.kx).reshape(self.N**2)
    w = w.reshape(self.N**2)
    jacobi = self.nu * self.L2 - \
            (np.diag(u[0]) @  self.Lx + np.diag(u[1]) @ self.Ly) - \
            (np.diag(self.Lx @ w) @ self.Ly - np.diag(self.Ly @ w) @ self.Lx) @ self.L2_inv
    return jacobi

  def dfds(self, w):
    # input: w is the vorticity in physical space, a 1D tensor of size N \times N,
    # output: \Delta w is the Laplacian of the vorticity in physical space, a 1D tensor of size N \times N,
    return self.L2 @ w.reshape(self.N**2)

  def CN(self, w_hat):
    # Crank-Nicolson scheme for spectral method with periodic boundary condition
    dt = self.dt
    nu = self.nu
    n = w_hat.shape[0]
    # the forcing is consistent with the choice of
    # https://arxiv.org/pdf/2010.08895
    # forcing = jnp.zeros_like(w_hat)
    # forcing = forcing.at[1, 1].set(n**2 / 2) * jnp.sin(self.t)
    """implementation 1: expansion method"""
    w_hat2 = jnp.zeros((n * 2, n + 1), dtype=jnp.complex128)
    psi_hat2 = jnp.zeros((n * 2, n + 1), dtype=jnp.complex128)
    w_hat2 = w_hat2.at[:n // 2, :n // 2 + 1].set(w_hat[:n // 2] * 4)
    w_hat2 = w_hat2.at[-n // 2:, :n // 2 + 1].set(w_hat[n // 2:] * 4)
    psi_hat2 = psi_hat2.at[:n // 2, :n // 2 +
                           1].set(-(w_hat / self.laplacian_)[:n // 2] * 4)
    psi_hat2 = psi_hat2.at[-n // 2:, :n // 2 +
                           1].set(-(w_hat / self.laplacian_)[n // 2:] * 4)
    wx2 = jnp.fft.irfft2(1j * w_hat2 * self.k2x)
    wy2 = jnp.fft.irfft2(1j * w_hat2 * self.k2y)
    psix2 = jnp.fft.irfft2(1j * psi_hat2 * self.k2x)
    psiy2 = jnp.fft.irfft2(1j * psi_hat2 * self.k2y)
    #print(np.linalg.norm(wx2*psiy2-wy2*psix2))
    tmp = jnp.zeros_like(w_hat, dtype=jnp.complex128)
    # k = 4
    # force = -k * jnp.cos(k * jnp.linspace(0, 2 * np.pi, n * 2, endpoint=False))
    tmp_ = jnp.fft.rfft2(wx2 * psiy2 - wy2 * psix2)  # + force[None])
    tmp = tmp.at[:n // 2].set(tmp_[:n // 2, :n // 2 + 1] / 4)
    tmp = tmp.at[n // 2:].set(tmp_[-n // 2:, :n // 2 + 1] / 4)
    """implementation 2: dealiasing method"""
    # w_hat_ = jnp.roll(w_hat, n // 2, axis=0)
    # w_hat_dealias = jnp.zeros_like(w_hat, dtype=jnp.complex128)
    # w_hat_dealias = w_hat_dealias.at[n // 2 - n // 3: n // 2 + n // 3, :n // 3].set(
    #   w_hat_[n // 2 - n // 3: n // 2 + n // 3, :n // 3]
    # )
    # w_hat_ = jnp.roll(w_hat_dealias, -n // 2, axis=0)
    # psi_hat = -(w_hat_ / self.laplacian_)
    # wx = jnp.fft.irfft2(1j * w_hat_ * self.kx)
    # wy = jnp.fft.irfft2(1j * w_hat_ * self.ky)
    # psix = jnp.fft.irfft2(1j * psi_hat * self.kx)
    # psiy = jnp.fft.irfft2(1j * psi_hat * self.ky)
    # tmp = jnp.fft.rfft2(wx * psiy - wy * psix)
    """implementation 3: pseudo spectral method"""
    # psi_hat = -(w_hat / self.laplacian_)
    # wx = jnp.fft.irfft2(1j * w_hat * self.kx)
    # wy = jnp.fft.irfft2(1j * w_hat * self.ky)
    # psix = jnp.fft.irfft2(1j * psi_hat * self.kx)
    # psiy = jnp.fft.irfft2(1j * psi_hat * self.ky)
    # tmp = jnp.fft.rfft2(wx * psiy - wy * psix)

    w_hat = ((1 + dt / 2 * nu * self.laplacian) * w_hat - dt * tmp) /\
      (1 - dt / 2 * nu * self.laplacian)# + dt * forcing
    return w_hat

  def CN_real(self, w):
    return jnp.fft.irfft2(self.CN(jnp.fft.rfft2(w[..., 0])))[..., None]

  def CN_adj(self, lambda_, i):
    # CN scheme for the dual variable \lambda
    print("This is an abstract f method!")
    raise NotImplementedError

  def set_x_hist(self, w, iter):
    step_num = self.step_num
    self.xhat_hist = np.zeros(
      (step_num, w.shape[0], w.shape[1]), dtype=np.complex128
    )
    self.u_hist = np.zeros((step_num, self.N, self.N, 2))
    self.ke = np.zeros(step_num)
    self.t = 0
    self.err_hist = np.zeros(step_num)
    # self.kappa = 4
    for i in range(step_num):
      self.xhat_hist[i] = w.copy()
      w = iter(w)
      self.t += self.dt
      #self.err_hist[i] = np.sum((irfft2(w) -
      #       2*self.kappa * np.cos(self.kappa*self.X)
      #       * np.cos(self.kappa*self.Y)
      #       * np.exp(
      #           np.Tensor([-2*self.kappa**2*(i+1)*self.dt*self.nu])))**2
      #         )
    from time import time
    t0 = time()
    # self.x_hist = np.array(np.fft.irfft2(self.xhat_hist, axes=(1, 2)))
    # psi = self.xhat_hist / self.laplacian_[None]
    # self.u_hist[..., 1] = -np.fft.irfft2(1j * psi * self.kx[None], axes=(1, 2))
    # self.u_hist[..., 0] = np.fft.irfft2(1j * psi * self.ky[None], axes=(1, 2))
    # self.e_kin = np.sum(self.u_hist**2, axis=(-3, -2, -1)) / 2
    self.x_hist = np.zeros((step_num, self.N, self.N))
    psi = self.xhat_hist / self.laplacian_[None]
    for i in range(step_num):
      self.x_hist[i] = jnp.fft.irfft2(self.xhat_hist[i])
      self.u_hist[i, ..., 1] = -jnp.fft.irfft2(1j * psi[i] * self.kx[None])
      self.u_hist[i, ..., 0] = jnp.fft.irfft2(1j * psi[i] * self.ky[None])
    self.e_kin = np.sum(self.u_hist**2, axis=(-3, -2, -1)) / 2
    print(f"Time taken: {time() - t0} seconds")


class ns_channel(dynamics):
  r"""
  
  $$
    \partial_t \bm{u} + (\bm{u} \cdot \nabla) \bm{u} = 
    -\nabla p + 1/\text{Re} \Delta \bm{u},    \\
    \nabla \cdot \bm{u} = 0, 	\\
    \bm{u} = \bm{0}, \quad y = 0, 1, \\
    \bm{u} = \bm{u_0}, \quad x = 0, \\
    \frac{\partial \bm{u}}{\partial n} = \bm{0}, \quad x = L_x, \\
  $$

  The dimension of the state variable is given by $N = 2 * n * (n/4)$ where
  $n, n/4$ is the spatial discretization size of x and y directions.
  """

  def __init__(
    self,
    model_type='NS',
    Lx=4,
    N=0,
    nx=128,
    ny=32,
    T=10,
    dt=0.01,
    Re=100,
    BC="Dirichlet",
    tol=1e-8,
    init_scale=4,
    tv_scale=1e-8,
    plot=False
  ):
    super(ns_channel,
          self).__init__(model_type, N, T, dt, tol, init_scale, tv_scale, plot)

    self.Lx = Lx
    self.nx = nx
    self.ny = ny
    self.dx = self.dy = self.Lx / self.nx
    self.N = (nx + 2) * (2 * ny + 3)
    self.dt = dt
    self.Re = Re
    self.BC = BC
    self.assembly_NSmatrix(self.nx, self.ny, self.dx, self.dy, self.BC)

  def assembly_NSmatrix(self, nx, ny, dx, dy, BC: str = "Dirichlet"):
    """assemble matrices used in the calculation
      LD: Laplacian operator with Dirichlet BC
      LN: Laplacian operator with Neuman BC, notice that this operator may have
      different form depends on the position of the boundary, here we use the
      case that boundary is between the outmost two grids
      L:  Laplacian operator associated with current BC with three Neuman BCs on
      upper, lower, left boundary and a Dirichlet BC on right
      """

    def Laplacian_Neumann(n):
      LN = jnp.roll(jnp.eye(n), 1, axis=1) + jnp.roll(jnp.eye(n), -1, axis=1) -\
        jnp.eye(n) * 2
      LN = LN.at[0, 0].set(-1)
      LN = LN.at[0, -1].set(0)
      LN = LN.at[-1, -1].set(-1)
      LN = LN.at[-1, 0].set(0)
      return LN

    LNx = Laplacian_Neumann(nx)
    LNy = Laplacian_Neumann(ny)
    L = jnp.kron(LNx /
                 (dx**2), jnp.eye(ny)) + jnp.kron(jnp.eye(nx), LNy / (dy**2))
    if BC == "Dirichlet":
      for i in range(ny):
        L = L.at[-1 - i, -1 - i].add(-2 / (dx**2))
    elif BC == "Neumann":
      L = jnp.vstack([L, jnp.ones_like(L[0:1])])
      L = jnp.hstack([L, jnp.ones_like(L[:, 0:1])])
      L = L.at[-1, -1].set(0)
    self.L = L

  def projection_correction(
    self,
    u: jnp.ndarray,
    v: jnp.ndarray,
    p: jnp.ndarray,
    t: float = 0,
    y0=0.325,
    eps=1e-7,
    dt=.01,
    correction: bool = False,
  ):
    """projection method to solve the incompressible NS equation
      The convection discretization is given by central difference
      u_ij (u_i+1,j - u_i-1,j)/2dx + \Sigma v_ij (u_i,j+1 - u_i,j-1)/2dx
      
    The collocation point of p locates at the center of the cell (1/2, 1/2)
    The collocation point of u locates at the right of p (1, 1/2)
    The collocation point of v locates at the top of p (1/2, 1)
    v[:, -1] = 0 for the no-slip boundary condition

    """

    def _u_padx(u: jnp.ndarray):
      return jnp.vstack([u_inlet, u, u[-1]])

    def _v_padx(v: jnp.ndarray):
      return jnp.vstack([2 * v_inlet - v[0], v, v[-1]])

    def _u_pady(u: jnp.ndarray):
      return jnp.hstack([-u[:, 0:1], u, -u[:, -1:]])

    def _v_pady(v: jnp.ndarray):
      return jnp.hstack([jnp.zeros_like(v[:, 0:1]), v, -v[:, -2:-1]])

    def _v2u(v: jnp.ndarray):
      """interpolate v to u"""
      v_pady = jnp.hstack([jnp.zeros_like(v[:, 0:1]), v])
      v = jnp.vstack([v_pady, v_pady[-1]])
      return (v[1:, 1:] + v[:-1, 1:] + v[1:, :-1] + v[:-1, :-1]) / 4

    def _u2v(u: jnp.ndarray):
      """interpolate u to v"""
      u_padx = jnp.vstack([u_inlet, u])
      u = jnp.hstack([u_padx, -u_padx[:, -1:]])
      return (u[1:, 1:] + u[:-1, 1:] + u[1:, :-1] + u[:-1, :-1]) / 4

    def grad_p(p: jnp.ndarray, dpdn: float = 0):
      """calculate the gradient of the pressure
      
      p: (nx, ny)
      dpdx: (nx - 1, ny)
      dpdy: (nx, ny - 1)
      dpdx[-1] = dpdy[:, -1] = 0 since p satisfies the Neuman BC 
      """
      if self.BC == "Dirichlet":
        p_padx = jnp.vstack([p, -p[-1:]])
      elif self.BC == "Neumann":
        p_padx = jnp.vstack([p, p[-1:] + dpdn * dx])
      dpdx = (p_padx[1:] - p_padx[:-1]) / dx
      dpdy = (p[:, 1:] - p[:, :-1]) / dy
      return dpdx, dpdy

    def div_uv(u: jnp.ndarray, v: jnp.ndarray):
      """calculate the divergence of the velocity field"""
      u_padx = jnp.vstack([u_inlet, u])
      dudx = (u_padx[1:] - u_padx[:-1]) / dx
      v_pady = jnp.hstack([jnp.zeros_like(v[:, 0:1]), v])
      dvdy = (v_pady[:, 1:] - v_pady[:, :-1]) / dy
      return dudx + dvdy

    def laplace_uv(u: jnp.ndarray, v: jnp.ndarray):
      """calculate the Laplacian of the velocity field"""

      u_padx = _u_padx(u)
      u_pad = _u_pady(u_padx)
      lapl_u = (u_pad[2:, 1:-1] - 2 * u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dx**2 +\
        (u_pad[1:-1, 2:] - 2 * u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dy**2
      v_padx = _v_padx(v)
      v_pad = _v_pady(v_padx)
      lapl_v = (v_pad[1:-1, 2:] - 2 * v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2]) / dy**2 +\
        (v_pad[2:, 1:-1] - 2 * v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1]) / dx**2
      return lapl_u, lapl_v

    def transport(u: jnp.ndarray, v: jnp.ndarray):
      """calculate the transport term of the velocity field"""
      u_padx = _u_padx(u)
      u_pady = _u_pady(u)
      uu_x = (u_padx[2:] - u_padx[:-2]) / dx / 2 * u
      vu_y = (u_pady[:, 2:] - u_pady[:, :-2]) / dy / 2 * _v2u(v)
      v_padx = _v_padx(v)
      v_pady = _v_pady(v)
      uv_x = (v_padx[2:] - v_padx[:-2]) / dx / 2 * _u2v(u)
      vv_y = (v_pady[:, 2:] - v_pady[:, :-2]) / dy / 2 * v

      return uu_x + vu_y, uv_x + vv_y

    def inlet(y: jnp.ndarray):
      """set the inlet velocity"""
      return y * (1 - y) * jnp.exp(-10 * (y - y0)**2)

    nx = self.nx
    ny = self.ny
    dx = self.dx
    dy = self.dy
    dt = self.dt
    Re = self.Re
    u_inlet = inlet(np.linspace(dy / 2, 1 - dy / 2, ny))
    v_inlet = inlet(np.linspace(dy, 1, ny)) * jnp.cos(t)

    dpdx, dpdy = grad_p(p)
    lapl_u, lapl_v = laplace_uv(u, v)
    du, dv = transport(u, v)
    u += -dpdx * dt
    v = v.at[:, :-1].add(-dpdy * dt)
    u += dt * (lapl_u / Re - du)
    v += dt * (lapl_v / Re - dv)
    v = v.at[:, -1].set(0)

    if not correction:
      # pressure correction
      res = div_uv(u, v) / dt
      if self.BC == "Dirichlet":
        p_res = jnp.linalg.solve(self.L, res.reshape(-1)).reshape([nx, ny])
        dpdn = 0
      elif self.BC == "Neumann":
        dpdn = -res.sum() / ny * dx
        res = res.at[-1].add(dpdn / dx)
        p_res = jnp.linalg.solve(
          self.L, jnp.hstack([res.reshape(-1), jnp.zeros(1)])
        )[:-1].reshape([nx, ny])

      dpdx, dpdy = grad_p(p_res, -dpdn)
      u += -dpdx * dt
      v = v.at[:, :-1].add(-dpdy * dt)
      p += p_res

    # res_ = div_uv(u, v)
    # if jnp.linalg.norm(res_) > eps:
    #   print(jnp.linalg.norm(res_))
    #   print("Velocity field is not divergence free!!!")

    return u, v, p

class Heat1D(dynamics):
  r"""1D Heat Equation: u_t = gamma * u_xx"""

  def __init__(
    self,
    model_type='Heat1D',
    L=10.0,
    N=256,
    T=5.0,
    dt=0.001,
    gamma=0.05,
    tol=1e-8,
    init_scale=1e-2,
    tv_scale=1e-8,
    rng=random.PRNGKey(0),
    plot=False,
    BC="periodic"
  ):
    super().__init__(model_type, N, T, dt, tol, init_scale, tv_scale, rng, plot)
    self.Lx = L
    self.gamma = gamma
    self.dx = L / N
    self.assembly_matrix()

  def assembly_matrix(self):
    N = self.N
    dx = self.dx
    # Periodic BC
    L2 = jnp.roll(jnp.eye(N), 1, axis=1) + jnp.roll(jnp.eye(N), -1, axis=1) - 2 * jnp.eye(N)
    L2 = L2 / dx**2
    self.L2 = L2

  def f(self, t, x):
    # Forward Euler RHS
    return self.gamma * self.L2 @ x

  def CN(self, x):
    # Crank-Nicolson time step
    dt = self.dt
    A = jnp.eye(self.N) - 0.5 * dt * self.gamma * self.L2
    B = jnp.eye(self.N) + 0.5 * dt * self.gamma * self.L2
    return solve(A, B @ x)