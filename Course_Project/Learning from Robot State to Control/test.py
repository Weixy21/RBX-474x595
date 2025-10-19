import torch
import scipy.io as sio
import models
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
plt.style.use('bmh')
from scipy.integrate import odeint

#choose a barriernet or not 1-barriernet
#barriernet = 1

#dynamics
def dynamics(y,t):
    dxdt = y[3]*np.cos(y[2])
    dydt = y[3]*np.sin(y[2])
    dttdt = y[4] #u1
    dvdt = y[5]  #u2
    return [dxdt,dydt,dttdt,dvdt, 0, 0]

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {} device".format(device))
torch.backends.cudnn.benchmark = True

train_data = sio.loadmat('data/dataM_train.mat') #data_train
train_data = train_data['data']
valid_data = sio.loadmat('data/dataM_valid.mat') 
valid_data = valid_data['data']
test_data = sio.loadmat('data/dataM_testn.mat')  # a new destination
test_data = test_data['data']

train0 = np.double(train_data[:,0:5])  # px, py, theta, v, dst
train_labels = np.reshape(np.double(train_data[:,5:7]), (len(train_data),2)) #theta_derivative, acc  4:6, 2
valid0 = np.double(valid_data[:,0:5]) 
valid_labels = np.reshape(np.double(valid_data[:,5:7]), (len(valid_data),2))
test0 = np.double(test_data[:,0:5]) 
test_labels = np.reshape(np.double(test_data[:,5:7]), (len(test_data),2))
init = test0[0]

mean = np.mean(train0, axis = 0)
std= np.std(train0, axis = 0)

train0 = (train0 - mean)/std
valid0 = (valid0 - mean)/std


# Initialize the model.
model_parameter = [] # todo

model = models.E2E_model(model_parameter, device, bn=False).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()    


px, py, theta, speed, dsty = init[0], init[1], init[2], init[3], init[4]
tr, tr0 = [], 0
ctrl1, ctrl2, safety, ctrl1_real, ctrl2_real, p1, p2, loc_x, loc_y = [], [], [], [], [], [], [], [], []
dt = [0,0.1]
obs_x, obs_y, R = 40, 15, 6
true_x, true_y = [], []

# running on a vehicle, closed-loop test
with torch.no_grad():
    for i in range(0,len(test0),1): #train0, 10
        #normalize
        x = (px - mean[0])/std[0]
        y = (py - mean[1])/std[1]
        tt = (theta - mean[2])/std[2]
        v = (speed - mean[3])/std[3]
        dst = (dsty - mean[4])/std[4]
        
        #get safety metric
        safe = (px - obs_x)**2 + (py - obs_y)**2 - R**2
        safety.append(safe)
        loc_x.append(px)
        loc_y.append(py)
        true_x.append(test0[i][0])
        true_y.append(test0[i][1])

        #prepare for model input
        x_r = Variable(torch.from_numpy(np.array([x,y,tt,v,dst])), requires_grad=False)
        x_r = torch.reshape(x_r, (1,nFeatures))
        x_r = x_r.to(device)
        ctrl = model(x_r, 0)
        
        
        #update state
        state = [px,py,theta,speed]

        state.append(ctrl[0,0].item()) 
        state.append(ctrl[0,1].item())
        
        #update dynamics
        rt = np.float32(odeint(dynamics,state,dt))
        px, py, theta, speed = rt[1][0], rt[1][1], rt[1][2], rt[1][3]
        

        ctrl1.append(ctrl[0,0].item()) 
        ctrl2.append(ctrl[0,1].item())
        ctrl1_real.append(test_labels[i][0])
        ctrl2_real.append(test_labels[i][1])
        tr.append(tr0)
        tr0 = tr0 + dt[1]
print("Implementation done!")


# Plot - todo
# Plot controls - ctrl1, ctrl2, ctrl1_real, ctrl2_real
# Plot robot trajectories - loc_x, loc_y, true_x, true_y, obstacle
# Plot safety metrics - safety

# plt.figure(1)
# todo
