import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x, dim=-1):
       
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, dim)


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
   
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  
    pdf = weights / torch.sum(weights, -1, keepdim=True)  
    cdf = torch.cumsum(pdf, -1) 
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  
  

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  
    

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def inference(model, xyz_, dir_, z_vals_, far,
              white_back, chunk, noise_std, weights_only=False):
  
    N_rays, N_samples = xyz_.shape[:2]
    rays_d_ = torch.repeat_interleave(dir_, repeats=N_samples, dim=0)  

    
    xyz_ = xyz_.view(-1, 3)  
    deltas = z_vals_[:, 1:] - z_vals_[:, :-1]  
    deltas = torch.cat([deltas, far - z_vals_[:, -1:]], -1)  


    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

   
    B = xyz_.shape[0]
    out_chunks = []
    for i in range(0, B, chunk):
        
        xyzdir = torch.cat([xyz_[i:i + chunk], rays_d_[i:i + chunk]], 1)
        rgb = model(xyzdir, sigma_only=False)
        out_chunks += [rgb]
    out_chunks = torch.cat(out_chunks, 0)

    if weights_only:
        sigmas = out_chunks.view(N_rays, N_samples)
    else:
        rgbsigma = out_chunks.view(N_rays, N_samples, 4)
        rgbs = rgbsigma[..., :3]  
        sigmas = rgbsigma[..., 3]  

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  

    T = torch.cumprod(alphas_shifted, -1)
    weights = alphas * T[:, :-1]  
    

    if weights_only:
        return weights
    else:
       
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  
        depth_final = torch.sum(weights * z_vals_, -1)  

        if white_back:
            weights_sum = weights.sum(1)  
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights
