import numpy as np
import torch, matplotlib.pyplot as plt
torch.manual_seed(7)
torch.set_default_dtype(torch.float32)
 
# ------------------ Known plant & helpers ------------------
Q, eta = 3600.0, 1.0
R0, R1, C1 = 0.05, 0.1, 100.0
tau = R1 * C1
Vmin, Vmax = 3.0, 4.2
 
def Ipulse(t, I0=1.0, period=200.0, duty=0.5):
    if not torch.is_tensor(t): t = torch.tensor(t, dtype=torch.float32)
    phase = torch.remainder(t, period) / period
    return (phase < duty).to(torch.float32) * I0
 
def OCV(z):  # simple affine SOC→voltage
    return Vmin + (Vmax - Vmin) * torch.clamp(z, 0.0, 1.0)
 
def rk4_integrate(f, y0, t):
    y_list=[y0]; yp=y0
    for i in range(len(t)-1):
        ti, h = t[i], t[i+1]-t[i]
        k1=f(ti, yp)
        k2=f(ti+0.5*h, yp+0.5*h*k1)
        k3=f(ti+0.5*h, yp+0.5*h*k2)
        k4=f(ti+h,     yp+h*k3)
        yp = yp + (h/6.0)*(k1+2*k2+2*k3+k4)
        y_list.append(yp)
    return torch.stack(y_list, 0)
 
# Ground-truth (only to synthesize a dataset for benchmarking)
def f_true(t, y):
    z, iR1 = y[...,0], y[...,1]
    I = Ipulse(t)
    dz   = -(eta/Q)*I
    diR1 = -(1.0/tau)*iR1 + (1.0/tau)*I
    return torch.stack([dz, diR1], dim=-1)
 
# ------------------ Make data ------------------
T, N = 800.0, 200
t = torch.linspace(0.0, T, N)
y0_true = torch.tensor([1.0, 0.0])
y_true  = rk4_integrate(f_true, y0_true, t)
z_true, iR1_true = y_true[:,0], y_true[:,1]
V_true = OCV(z_true) - R0*Ipulse(t) - R1*iR1_true
V_true = V_true + 0.01*torch.randn_like(V_true)  # noise
 
n_train = int(0.8*N)
idx_fit = torch.arange(0, n_train-1)  # training intervals
