"""
1. write a model wrapper that wraps the vae encoded tensor with DiagonalGaussianDistribution
2. write pixel-wise sampling from the distribution
"""

import torch

class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        noise = torch.randn_like(self.std)
        return self.mean + self.std * noise

    def mode(self):
        return self.mean


if __name__ == "__main__":
    latent_dist = DiagonalGaussianDistribution(torch.randn(1, 2, 1, 1))
    print(latent_dist.sample())