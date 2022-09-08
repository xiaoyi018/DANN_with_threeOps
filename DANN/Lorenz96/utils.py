import numpy as np
import torch
import parameters

class L96_Integrator:
    def __init__(self, tp=1):
        self.J      = parameters.J
        self.F      = parameters.F
        self.dt     = parameters.step_size * tp

    def _odeL96(self, xin, noise=False):
        if type(xin) == torch.Tensor:
            dpred = xin.clone()
        elif type(xin) == np.ndarray:
            dpred = xin.copy()
        else:
            raise NotImplementedError("not implemented")
            
        dpred[0] = (xin[1] - xin[self.J-2]) * xin[self.J-1] - xin[0]
        dpred[1] = (xin[2] - xin[self.J-1]) * xin[0] - xin[1]
        dpred[self.J-1] = (xin[0] - xin[self.J-3]) * xin[self.J-2] - xin[self.J-1]
        for j in range(2, self.J-1):
            dpred[j] = (xin[j+1] - xin[j-2]) * xin[j-1] - xin[j]
        dpred = dpred + self.F

        if noise:
            dpred += parameters.background_err

        return dpred
    
    def _RK4Solver(self, x, ):
        k1 = self._odeL96(x)
        x2 = x + 0.5 * self.dt * k1
        k2 = self._odeL96(x2)

        x3 = x + 0.5 * self.dt * k2
        k3 = self._odeL96(x3)

        x4 = x + self.dt * k3
        k4 = self._odeL96(x4)

        return x + self.dt * (k1+2.*k2+2.*k3+k4)/6.
    
    def _NoiseSolver(self, x):
        k1 = self._odeL96(x, True)
        x2 = x + 0.5 * self.dt * k1
        k2 = self._odeL96(x2, True)

        x3 = x + 0.5 * self.dt * k2
        k3 = self._odeL96(x3, True)

        x4 = x + self.dt * k3
        k4 = self._odeL96(x4, True)
        return x + self.dt * (k1+2.*k2+2.*k3+k4)/6.
    
    def _EulerSolver(self, x):
        k1 = self._odeL96(x)
        x2 = x + self.dt * k1
        k2 = self._odeL96(x2)
        return x + self.dt * (k1+k2)/2.

    def forward(self, x, i, mean=0.0, std=1.0):
        X = std * x 
        X = X + mean

        if i == 1:
            xpred = self._RK4Solver(X)
        elif i == 2:
            xpred = self._EulerSolver(X)
        else:
            xpred = self._NoiseSolver(X)

        xpred = xpred - mean
        xpred = xpred / std
        return xpred

def dynamic_loss(x, n, mean, std, tp=1):
    integrator = L96_Integrator()
    xx = x.transpose(0,1)
    for i in range(n):
        xx = integrator.forward(xx, tp, mean=torch.Tensor(mean), std=torch.Tensor(std))
    x_pred = xx.transpose(0,1)
    return torch.mean((x[:,:,n:] - x_pred[:,:,:-n])**2 )