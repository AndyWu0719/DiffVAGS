import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from typing import TypeVar, List
Tensor = TypeVar('torch.tensor')


# Gaussian VAE Encoder
class GaussianEncoder(nn.Module):
    # num_iterations = 0

    def __init__(self, 
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 kl_std=1.0,
                 beta: int = 4,
                 gamma: float = 10.0,
                 max_capacity: int = 25,
                 capacity_max_iteration: int = 1e5,
                 loss_type: str = "B",
                 **kwargs) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 512, 512]
        self.hidden_dims = hidden_dims
        
        self.kl_std = kl_std
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.max_capacity = max_capacity
        self.capacity_max_iteration = capacity_max_iteration

        self.register_buffer('C_max', torch.tensor([self.max_capacity], dtype=torch.float32))
        
        
        print(f"Initializing GaussianEncoder:")
        print(f"  Input channels: {self.in_channels}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Hidden dims: {self.hidden_dims}")
        print(f"  Loss type: {self.loss_type}")

        self.C_max = torch.Tensor([self.max_capacity])
        self.capacity_max_iteration = self.capacity_max_iteration


        # ------Encoder------]
        encoder_modules = []
        current_channels = self.in_channels

        for h_dim in self.hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            current_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        feature_dim = self.hidden_dims[-1] * 4
        
        self.fc_mu = nn.Linear(feature_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, self.latent_dim)
        
        # ------Decoder------
        decoder_modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * 4)

        decoder_hidden_dims = self.hidden_dims[::-1]
        
        for i in range(len(decoder_hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_hidden_dims[i],
                                     decoder_hidden_dims[i + 1],
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
                    nn.BatchNorm2d(decoder_hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*decoder_modules)

        # Final layer: 64x64 -> in_channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(decoder_hidden_dims[-1],
                             decoder_hidden_dims[-1],
                             kernel_size=3,
                             stride=2,
                             padding=1,
                             output_padding=1),
            nn.BatchNorm2d(decoder_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(decoder_hidden_dims[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, encode_input) -> List[Tensor]:
        # Pass through encoder
        result = self.encoder(encode_input)  # [B, hidden_dims[-1], 2, 2]
        result = torch.flatten(result, start_dim=1)  # [B, hidden_dims[-1]*4]
        
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)

        return [mu, logvar]
    
    def decode(self, z: Tensor) -> Tensor:
        # decode the latent variable z
        result = self.decoder_input(z)  # [B, hidden_dims[-1]*4]
        result = result.view(-1, int(result.shape[-1]/4), 2, 2)  # [B, hidden_dims[-1], 2, 2]
        result = self.decoder(result)
        result = self.final_layer(result)  # [B, C, 64, 64]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, data: Tensor, multiview_images: Tensor = None, **kwargs) -> Tensor:
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return [recon_x, data, mu, logvar, z]
    
    def loss_function(self, *args, **kwargs) -> dict:
        # Placeholder for loss function logic
        # self.num_iterations += 1
        
        kld_weight = kwargs.get('M_N', 1.0)  # minibatch weight
        
        recons, data, mu, logvar, z = args
        kl_loss = self._compute_kl_loss(mu, logvar)
        vae_loss = kld_weight * kl_loss
        
        return {
            'VAEloss': vae_loss,
            'kld_unweighted': kl_loss
        }
        
    def _compute_kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:

        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, logvar)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            return l2_size_loss / latent.shape[0]
        else:
            std = torch.exp(0.5 * logvar)
            zeros = torch.zeros_like(mu)
            kl_std_tensor = torch.ones_like(std) * self.kl_std
            gt_dist = torch.distributions.normal.Normal(zeros, kl_std_tensor)
            sampled_dist = torch.distributions.normal.Normal(mu, std)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist)
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()
            return kl_loss

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        # Generate samples from the latent space and return the images
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        # Generate images from the input image
        return self.forward(x)[0]
    
    def get_latent(self, x: Tensor, **kwargs) -> Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z