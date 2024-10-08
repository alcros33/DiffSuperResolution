import math
import torch
import torch.nn as nn

def get_from_idx(element: torch.Tensor, idx: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, idx)
    return ele.reshape(-1, 1, 1, 1)

class ResShiftDiffusion(nn.Module):
    def __init__(self, timesteps=15, p=0.3, kappa=2.0, etas_end=0.99, min_noise_level=0.04):
        super().__init__()
        self.timesteps = timesteps
        self.kappa = kappa
        sqrt_eta_1  = min(min_noise_level / kappa, min_noise_level)
        b0 = math.exp(1/float(timesteps-1)*math.log(etas_end/sqrt_eta_1))
        base = torch.ones(timesteps)*b0
        beta = ((torch.linspace(0,1,timesteps))**p)*(timesteps-1)
        sqrt_eta = torch.pow(base, beta) * sqrt_eta_1

        self.register_buffer("sqrt_eta", sqrt_eta)
        self.register_buffer("eta", sqrt_eta**2)

        prev_eta = torch.roll(self.eta, 1)
        prev_eta[0] = 0

        alpha = self.eta - prev_eta
        self.register_buffer("alpha", alpha)
        
        self.register_buffer("backward_mean_c1", prev_eta / self.eta)
        self.register_buffer("backward_mean_c2", self.alpha / self.eta)
        backward_variance =  kappa*kappa*prev_eta*self.alpha/self.eta
        backward_variance[0] = backward_variance[1]
        # self.register_buffer("backward_variance", backward_variance)
        self.register_buffer("backward_std", torch.exp(0.5*torch.log(backward_variance)))
    
    
    def add_noise(self, x, y, epsilon, t):
        eta = get_from_idx(self.eta, t)
        sqrt_eta = get_from_idx(self.sqrt_eta, t)
        
        mean = x + eta*(y-x)
        std = self.kappa*sqrt_eta

        return mean + std*epsilon
    
    def backward_step(self, x_t, predicted_x0, t):
        mean_c1, mean_c2 = get_from_idx(self.backward_mean_c1, t), get_from_idx(self.backward_mean_c2, t)
        std = get_from_idx(self.backward_std, t)

        mean = mean_c1*x_t + mean_c2*predicted_x0

        mask = (t!=0).float().reshape(-1, 1, 1, 1)

        return mean + mask*std*torch.randn_like(mean)
    
    def prior_sample(self, y, epsilon):
        t = torch.tensor([self.timesteps-1,] * y.shape[0], device=y.device).long()
        return y + self.kappa * get_from_idx(self.sqrt_eta, t) * epsilon


class ResShiftDiffusionEps(ResShiftDiffusion):
    def __init__(self, timesteps=15, p=0.3, kappa=2.0, etas_end=0.99, min_noise_level=0.04):
        super().__init__(timesteps=timesteps, p=p, kappa=kappa, etas_end=etas_end, min_noise_level=min_noise_level)
        self.register_buffer("x0_scale", 1.0/(1-self.eta))
        self.x0_scale[0] = 1.0
    
    def backward_step(self, x_t, y_0, predicted_eps, t):
        sqrt_eta = get_from_idx(self.sqrt_eta, t)
        eta = get_from_idx(self.eta, t)
        x0_scale = get_from_idx(self.x0_scale, t)

        predicted_x0 = x0_scale*(x_t - eta*y_0  - self.kappa*sqrt_eta*predicted_eps)

        mean_c1, mean_c2 = get_from_idx(self.backward_mean_c1, t), get_from_idx(self.backward_mean_c2, t)
        std = get_from_idx(self.backward_std, t)

        mean = mean_c1*x_t + mean_c2*predicted_x0

        return mean + std*torch.randn_like(x_t)


class ResShiftDiffusionEpsDumb(ResShiftDiffusion):
    def __init__(self, timesteps=15, p=0.3, kappa=2.0, etas_end=0.99, min_noise_level=0.04):
        super().__init__(timesteps=timesteps, p=p, kappa=kappa, etas_end=etas_end, min_noise_level=min_noise_level)
        self.register_buffer("x0_scale", 1.0/(1-self.eta))
    
    def backward_step(self, x_t, y_0, predicted_eps, t):
        sqrt_eta = get_from_idx(self.sqrt_eta, t)
        eta = get_from_idx(self.eta, t)

        predicted_x0 = self.x0_scale*(x_t - eta*y_0  - self.kappa*sqrt_eta*predicted_eps)

        mean_c1, mean_c2 = get_from_idx(self.backward_mean_c1, t), get_from_idx(self.backward_mean_c2, t)

        mean = mean_c1*x_t + mean_c2*predicted_x0

        return mean


class SimpleDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.timesteps = timesteps
        # The betas and the alphas
        self.register_buffer("beta" ,self.get_betas(beta_schedule, beta_start, beta_end))
        # Some intermediate values we will need
        self.register_buffer("alpha", 1 - self.beta)
        # self.register_buffer("alpha", 1 - torch.cat([torch.zeros(1).to(self.beta.device), self.beta], dim=0))
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer("one_by_sqrt_alpha", 1. / torch.sqrt(self.alpha))
        self.register_buffer("sqrt_one_minus_alpha_bar",torch.sqrt(1 - self.alpha_bar))

    def get_betas(self, beta_schedule, beta_start, beta_end):
        if beta_schedule == "linear":
            return torch.linspace(
                beta_start,
                beta_end,
                self.timesteps,
                dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            return torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                self.timesteps,
                dtype=torch.float32) ** 2

    def forward(self, x0: torch.Tensor, timesteps: torch.Tensor, epsilon):
        # Generate normal noise
        mean    = get_from_idx(self.sqrt_alpha_bar, timesteps) * x0      # Mean
        std_dev = get_from_idx(self.sqrt_one_minus_alpha_bar, timesteps) # Standard deviation
        # Sample is mean plus the scaled noise
        sample  = mean + std_dev * epsilon
        return sample
    
    def backward_step(self, xnoise: torch.Tensor, timestep:int, predicted_noise:torch.Tensor):
        # Noise from normal distribution
        z  = torch.randn_like(xnoise) if timestep > 0 else torch.zeros_like(xnoise)
        beta_t                     = self.beta[timestep].reshape(-1, 1, 1, 1)
        one_by_sqrt_alpha_t        = self.one_by_sqrt_alpha[timestep].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[timestep].reshape(-1, 1, 1, 1)
        # Use the formula above to sample a denoised version from the noisy one
        xdenoised = (
            one_by_sqrt_alpha_t
            * (xnoise - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )
        return xdenoised
