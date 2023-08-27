# This script plot \cref{RD-ds}
# This script plot \cref{NS-ds}
import argparse
import numpy.random as r
from utils import *
from simulator import *
from models import *
from matplotlib import pyplot as plt
from matplotlib import cm


r.seed(0)
n, beta, Re, type, GPU, ds_parameter = parsing()
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

arg, u, v, label = read_data('../../data/{}/{}-{}.npz'.format(type, n, ds_parameter))

nx, ny, dt, T, label_dim, traj_num, step_num, test_index, u, v, label = preprocessing(arg, type, u, v, label, device, flag=False)

print('Plotting {} model with n = {}, beta = {:.1f}, Re = {} ...'.format(type, n, beta, Re))
    

if type == 'NS':
    model_ols = UNet([2,4,8,16,32,64,1]).to(device)
    u64_np = copy.deepcopy(u[test_index].numpy()).reshape([step_num, nx+2, ny+2])
    v64_np = copy.deepcopy(v[test_index].numpy()).reshape([step_num, nx+2, ny+1])
else:
    model_ols = UNet().to(device)
    u64_np = copy.deepcopy(u[test_index].numpy()).reshape([step_num, nx, ny])
    v64_np = copy.deepcopy(v[test_index].numpy()).reshape([step_num, nx, ny])
model_ols.load_state_dict(torch.load('../../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter), 
                                    map_location=torch.device('cpu')))
model_ols.eval()
model_ed = EDNet().to(device)
model_ed.load_state_dict(torch.load('../../models/{}/ED-{}-{}.pth'.format(type, n, ds_parameter),
                                    map_location=torch.device('cpu')))
model_ed.eval()


if type == 'RD':
    simulator_ols = RD_Simulator(model_ols, 
                                 model_ed, 
                                 device, 
                                 u_hist=u64_np, 
                                 v_hist=v64_np, 
                                 step_num=step_num,
                                 dt=dt)
else:
    simulator_ols = NS_Simulator(model_ols, 
                                 model_ed, 
                                 device, 
                                 u_hist=u64_np, 
                                 v_hist=v64_np, 
                                 step_num=step_num,
                                 dt=dt)
simulator_ols.simulator()  


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi']  = 300 
width = 10
height = 4
#################################################################
#               Plot of the simulation result                   #
#################################################################
fig = plt.figure(figsize=(width, height))
fig_num = 5
hspace = 0.01
wspace = -0.35
fraction = 0.024
pad = 0.001
t_array = (np.array([0.0, 0.2, 0.4, 0.6, 1.0])*90).astype(int)
print(t_array)
ax1 = []
ax2 = []
for i in range(fig_num):
    ax1.append(fig.add_subplot(2,fig_num,i+1))
    ax2.append(fig.add_subplot(2,fig_num,i+fig_num+1))
fig.tight_layout()
for i in range(fig_num):
    ax1[i].imshow(simulator_ols.u_hist[t_array[i]], cmap=cm.jet)
    ax1[i].set_axis_off()
    im = ax2[i].imshow(simulator_ols.u_hist_simu[t_array[i]], cmap=cm.jet)
    ax2[i].set_axis_off()
plt.subplots_adjust(hspace=hspace)
plt.subplots_adjust(wspace=wspace)
for i in range(fig_num):
    cbar = fig.colorbar(im, ax=[ax1[i], ax2[i]], fraction=fraction, pad=pad, orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
plt.savefig('../../fig/exp1/{}/1.jpg'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.clf()


#################################################################
#              Plot of the ds & error comparison                #
#################################################################
fig = plt.figure(figsize=(width, height//2))
ax = fig.add_subplot(111)
ax.plot(np.linspace(0, 800, int(simulator_ols.error_hist.shape[0])), simulator_ols.error_hist/10, label='OLS error', color='r', linewidth=.5)
ax.legend(bbox_to_anchor = (0.1, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=10)
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)
ax.set_ylabel('error', fontsize=10)
ax1 = ax.twinx()
ax1.plot(np.linspace(0, 800, int(simulator_ols.ds_hist.shape[0])), np.log10(simulator_ols.ds_hist[::-1])-3, label='OLS log(ds)', color='r', linewidth=.5, linestyle='dashed')
ax1.legend(bbox_to_anchor = (0.1, 0.7), loc = 'upper left', borderaxespad = 0., fontsize=10)
ax1.set_ylabel('log of dds', fontsize=10)
plt.savefig('../../fig/exp1/{}/2.jpg'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0)
#plt.show()


# For the comparison bewteen RD & NS, we need to show following features of our figures
# 1. Two figures should be of the same time scope: same \Delta t & same number of step_num
# 2. RD should suffer from less severe distribution shift comparing to NS, meaning that 
# both the error and ds should be smaller than NS.
# 3. Currently, the ds legend in NS is too small.