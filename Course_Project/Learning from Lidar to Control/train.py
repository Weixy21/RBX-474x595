import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from cvxopt import solvers, matrix
import copy
from qpth.qp import QPFunction, QPSolvers
import pickle
from torch.nn.functional import normalize

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri8', 'adams'], default='dopri8')   #dopri5
parser.add_argument('--activation', type=str, choices=['gelu', 'silu', 'tanh'], default='gelu') 
parser.add_argument('--data_size', type=int, default=100)   # length of each trajectory
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20) 
parser.add_argument('--niters', type=int, default=200)  #2000
parser.add_argument('--test_freq', type=int, default=2)  #20
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--consider_noise', action='store_true')
parser.add_argument('--num_var', type=int, default=1)  # number of parameters in the neural ODE chosen (to be modified) to enforce the invariance
parser.add_argument('--use_cnn', action='store_true')
parser.add_argument('--use_fc', action='store_true')
parser.add_argument('--use_diff_weight', action='store_true')  # for loss

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


exp_id = '0'  

t = torch.linspace(0., 9.9, args.data_size).to(device)

pkl_file = open('./data/data4.pkl','rb')
data = pickle.load(pkl_file)
lidar = torch.tensor(data['lidar']).float()  # batch, seq, data
control = torch.tensor(data['ctrl']).float()
ego = torch.tensor(data['ego']).float()
other = torch.tensor(data['other']).float()
pkl_file.close()

# Training data
lidar_bat = lidar.unsqueeze(2).to(device)   # batch, seq, data_h x data_w
control_bat = control.unsqueeze(2).to(device)
ego_bat = ego.unsqueeze(2).to(device)
other_bat = other.unsqueeze(2).to(device)
lidar_bat = lidar_bat/200.  #normalize lidar
ego_bat[:,:,:,3] = ego_bat[:,:,:,3]/180 # normalize speed

#Testing data
lidar_test = lidar_bat[-1,:,:,:]
control_test = control_bat[-1,:,:,:]
ego_test = ego_bat[-1,:,:,:]
other_test = other_bat[-1,:,:,:]

# initial condition for closed-loop testing
control_test0 = control_bat[-1,0,:,:]
lidar_test0 = lidar_bat[-1,0,:,:]
ego_test0 = ego_bat[-1,0,:,:]

def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    return sol['x']


def get_batch(bitr):  # get training data from trajectory bitr
    control = control_bat[bitr,:,:,:]
    lidar = lidar_bat[bitr,:,:,:]
    ego = ego_bat[bitr,:,:,:]
    bb = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size-1, replace=False))
    aa = torch.tensor([0])   # augment 0
    s = torch.cat([bb, aa], dim = 0)
    batch_u0 = control[s]  # (M, D)
    batch_lidar0 = lidar[s]
    batch_ego0 = ego[s]
    batch_t = t[:args.batch_time]  # (T)
    batch_u = torch.stack([control[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    batch_lidar = torch.stack([lidar[s + i] for i in range(args.batch_time)], dim=0)
    batch_ego = torch.stack([ego[s + i] for i in range(args.batch_time)], dim=0)
    return batch_u0.to(device), batch_lidar0.to(device), batch_ego0.to(device), batch_t.to(device), batch_u.to(device), batch_lidar.to(device), batch_ego.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    if args.consider_noise:
        makedirs('png_noise')
    else:
        makedirs('png_cbf' + exp_id)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4), facecolor='white')
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_phase = fig.add_subplot(122, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('u1,u2')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('u1')
        ax_phase.set_ylabel('u2')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        # ax_phase.set_xlim(-2, 2)
        # ax_phase.set_ylim(-2, 2)

        fig.tight_layout()
        if args.consider_noise:
            plt.savefig('png_noise/{:03d}'.format(itr))
        else:
            plt.savefig('png_cbf' + exp_id +'/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)

################################################## neural ODE model
class ODEFunc(nn.Module):

    def __init__(self, fc_param, conv_filters,
                 dropout= 0.2):
        super(ODEFunc, self).__init__()
	
	# build your model here

        # self.net = self.build_mlp(fc_param) # todo

        self.use_dcbf = False
        self.lidar = lidar_test0
        self.ego = ego_test0
        if args.use_cnn == True:
            self._cnn = self._build_cnn(conv_filters, dropout=dropout)
        
    def forward(self, t, y):

 	# todo
        # out = []
                        
        return out

  
    def _build_cnn(self, filters, dropout=0., no_act_last_layer=False):
        
        modules = nn.ModuleList()
        for i, filt in enumerate(filters):
            modules.append(nn.Conv1d(*filt))
            if (i != len(filters) - 1) or (not no_act_last_layer):
                modules.append(nn.BatchNorm1d(filt[1]))
                modules.append(nn.ReLU())
                if dropout > 0:
                    modules.append(nn.Dropout(p=dropout))
        modules = nn.Sequential(*modules)
        
        return modules
    
    def build_mlp(self, filters, dropout=0.2, no_act_last_layer=True, activation='gelu'):
        if activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError(f'Not supported activation function {activation}')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not (no_act_last_layer and i == len(filters)-2):
                modules.append(activation)
                # if dropout > 0.:
                #     modules.append(nn.Dropout(p=dropout))

        modules = nn.Sequential(*modules)
        return modules


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0
    if args.consider_noise:
        makedirs('models_noise')
    else:
        makedirs('models_cbf' + exp_id)
    
    if args.use_cnn or args.use_fc:
        fc_param = [2, 64, 128, 30]  # [2, 32, 64, 26]   # +ego speed, ego heading
    else:
        # fc_param = [2, 50, 400, 206]
        fc_param = [2, 64, 256, 512, 206]
    #channel x output channels x kernel size x stride x padding
    conv_param = [[1, 4, 5, 2, 1], [4, 8, 3, 2, 1], [8, 12, 3, 2, 0]]
    func = ODEFunc(fc_param, conv_param).to(device)  # initialize the model
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    test_loss = torch.tensor([0]).to(device)
    tt = torch.tensor([0., 0.1]).to(device)
    for itr in range(1, args.niters + 1):         # training loop
        for bitr in range(200):  # 200 trajectories
            func.use_dcbf = False
            func.train()
            optimizer.zero_grad()
            batch_u0, batch_lidar0, batch_ego0, batch_t, batch_u, batch_lidar, batch_ego = get_batch(bitr)
            u0 = batch_u0
            pred_u = u0.unsqueeze(0)
            for i in range(args.batch_time-1):
                print('iteration: ', itr, '| traj: ', bitr, '| step:', i, 'test loss: ', test_loss.item())
                batch_lidar_i = batch_lidar[i,:,:,:]
                if args.use_cnn == True:
                    sensor_cnn = func._cnn(batch_lidar_i)
                    sensor_max = sensor_cnn.max(dim=-1)[0]
                    func.lidar = sensor_max.unsqueeze(1)
                else:
                    func.lidar = batch_lidar_i
                func.ego = batch_ego[i,:,:,:] 
                pred = odeint(func, u0, tt).to(device)   # , method = 'explicit_adams' euler, midpoint, rk4, explicit_adams, fixed_adams
                #"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams", "fixed_adams", "scipy_solver"
                pred_u = torch.cat([pred_u, pred[-1,:,:,:].unsqueeze(0)], dim = 0)
                u0 = pred[-1,:,:,:]
            
            if args.use_diff_weight:
                loss = 0.01*torch.mean(torch.abs(pred_u[:,:,:,0] - batch_u[:,:,:,0])) + 0.99*torch.mean(torch.abs(pred_u[:,:,:,1] - batch_u[:,:,:,1]))
            else:
                loss = torch.mean(torch.abs(pred_u - batch_u))
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())


        if itr % args.test_freq == 0:
            with torch.no_grad():
                func.use_dcbf = False
                u0 = control_test0
                pred_u = u0.unsqueeze(0)
                func.eval()
                for i in range(args.data_size-1):
                    lidar_test_i = lidar_test[i,:,:]
                    if args.use_cnn == True:
                        x = lidar_test_i.unsqueeze(0)
                        sensor_cnn = func._cnn(x)
                        func.lidar = sensor_cnn.max(dim=-1)[0]
                    else:
                        func.lidar = lidar_test_i
                    func.ego = ego_test[i,:,:] 
                    pred = odeint(func, u0, tt)
                    pred_u = torch.cat([pred_u, pred[-1,:,:].unsqueeze(0)], dim = 0)
                    u0 = pred[-1,:,:]
                if args.use_diff_weight:
                    test_loss = 0.01*torch.mean(torch.abs(pred_u[:,:,0] - control_test[:,:,0])) + 0.99*torch.mean(torch.abs(pred_u[:,:,1] - control_test[:,:,1]))
                else:
                    test_loss = torch.mean(torch.abs(pred_u - control_test))

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, test_loss.item()))
                visualize(control_test, pred_u, ii)
                ii += 1
            if args.consider_noise:
                torch.save(func.state_dict(), "./models_noise/model_node_itr" + format(ii, '02d') + ".pth")
            else:
                torch.save(func.state_dict(), "./models_cbf"+ exp_id + "/model_node_itr" + format(ii, '02d') + ".pth")
            
        end = time.time()

    # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
    # continue
