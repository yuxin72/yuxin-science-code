import torch.nn as nn 
import torch.optim as optim


# Batching algorithm 

batch_time = 50
data_size = 200

def get_batch():
    # random starting indice 
    s = np.random.randint(0, data_size - batch_time)

    # initial condition for each trajectory
    batch_y0 = y_true[s]  # [z, iR1]
    batch_t0 = t[s] 
    
    # time vector
    batch_t = t[s: s+batch_time+1]  # [normalised batch_time]
    
    # targets: for each start index, grab the next batch_time states
    batch_targets = V_true[s:s+batch_time+1]  # [batch_size]

    return batch_y0, batch_t.float(), batch_targets

#Creating Neural ODE model 

class Thev_model(nn.Module):
    def __init__(self, current_profile):
        super(Thev_model, self).__init__()
        self.current_profile = current_profile
        
        self.log_Vmax = torch.log(torch.tensor(4.2)) 
        self.log_Vmin = torch.log(torch.tensor(3.0)) 
        self.log_R0 = torch.log(torch.tensor(0.05)) 
        self.log_R1 = torch.log(torch.tensor(0.1)) 

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, 2),
        ) 
            
    def forward(self, t, y):
        '''
        y is [2] i.e. [[z, IR1], ...]
        output is [dzdt, dIR1dt] 
        ''' 
        dy = self.net(y)
        
        return dy
#Train
num_iters = 500
func = Thev_model(Ipulse)
freq = 50 
t_eval = t
x0 = y0_true 

optimiser = optim.Adam(func.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()  

loss_values = []
iter_values = []

for i in range(num_iters):
    optimiser.zero_grad()
    y0s, t_batch, targets = get_batch()
    pred_ys = odeint(func, y0s, t_batch, rtol=1e-5) #[times i.e. 40, state dim] 
    z, iR1  = pred_ys[:,0], pred_ys[:, 1]
    
    R0 = torch.exp(func.log_R0)
    R1 = torch.exp(func.log_R1)
    Vmin = torch.exp(func.log_Vmin)
    Vmax = torch.exp(func.log_Vmax)

    ocv = Vmin + (Vmax - Vmin)*z #[times, batch size]
    term_v = ocv - R0*func.current_profile(t_batch) - R1*iR1 #[times, batch size] 

    loss = loss_fn(term_v, targets)
    loss.backward()
    optimiser.step()

    if i%freq == 0:
        with torch.no_grad():
            pred_y = odeint(func, x0, t_eval, rtol = 1e-5) #[times, state dim] #uses adjoint to obtain gradients, now get the solution at all time points
            z, iR1 = pred_y[:, 0], pred_y[:,1]

            R0 = torch.exp(func.log_R0)
            R1 = torch.exp(func.log_R1)
            Vmin = torch.exp(func.log_Vmin)
            Vmax = torch.exp(func.log_Vmax)

            ocv = Vmin + (Vmax - Vmin)*z #[times, batch size]
            print(len(z))
            term_v_s = ocv - R0*func.current_profile(t_eval) - R1*iR1 #[times, batch size]
    
            loss = loss_fn(term_v_s, V_true)
            loss_values.append(loss)
            iter_values.append(i)
            print(f"Iter: {i}, loss: {loss.item()}")

            plt.plot(t_eval, V_true, 'g-', label='Truth')
            plt.plot(t_eval, term_v_s, 'b--', label='Prediction')
            plt.xlabel('t')
            plt.ylabel('v(t)')
            plt.title('Graph of terminal voltage against time')
            plt.legend()
    
            plt.show()

loss_values = np.array(loss_values)
iter_values = np.array(iter_values)
plt.plot(iter_values, loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
            
