import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from typing import TypeVar, List
Tensor = TypeVar('torch.tensor')


# Gaussian VAE Encoder
class GaussianEncoder(nn.Module):
    """
    Gaussian VAE Encoder using MobileNetV3 architecture.
    Args:
        in_channels (int): Number of input channels.
        latent_dim (int): Number of latent variables.
        hidden_dims (List): List of hidden dimensions for the encoder.
        kl_std (float): Standard deviation for KL divergence.
        beta (int): Weight for the KL divergence term.
        gamma (float): Weight for the reconstruction loss.
        max_capacity (int): Maximum capacity for the KL divergence.
        capacity_max_iteration (int): Maximum number of iterations for capacity.
        loss_type (str): Type of loss function to use.

    Input: mutlimodal data
    Shape: [batch_size, in_channels, 64, 64]
    Output: [recon_x, data, mu, logvar, z]    

    recon_x: [batch_size, in_channels, 64, 64]
    data: [batch_size, in_channels, 64, 64]
    mu: [batch_size, latent_dim]
    logvar: [batch_size, latent_dim]
    z: [batch_size, latent_dim]
    """

    def __init__(self, 
                 in_channels: int = 128,
                 latent_dim: int = 768,
                 hidden_dims: List = [16, 24, 40],
                 kl_std=1.0,
                 beta: int = 4,
                 gamma: float = 10.,
                 max_capacity: int = 25,
                 capacity_max_iteration: int = 1e5,  # 10000 in default configs
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super().__init__()
        self.num_iterations = 0
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.kl_std = kl_std
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.register_buffer('capacity_max', torch.tensor([max_capacity], dtype=torch.float32))
        self.capacity_stop_iteration = capacity_max_iteration
        # MobileNetV3
        self.hidden_dims = hidden_dims

        # ------Encoder------
        # Layer 1: stride=2, downsample
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.Hardswish(),
        )
        # Layer 2, 3: simple MobileNetV3 blocks
        self.encoder.add_module("mbconv1", self._mobilenetv3_block(hidden_dims[0], hidden_dims[1], stride=2))
        self.encoder.add_module("mbconv2", self._mobilenetv3_block(hidden_dims[1], hidden_dims[2], stride=2))
        # Layer 4, 5: input: 64x64 -> output: 8x8, Fully connected
        encoder_out_size = 8
        fc_input_dim = hidden_dims[-1] * encoder_out_size * encoder_out_size
        self.fc_mu = nn.Linear(fc_input_dim, latent_dim)    # Mean
        self.fc_logvar = nn.Linear(fc_input_dim, latent_dim)   # Log Variance
        
        # ------Decoder------
        # latent_dim -> fc_input_dim
        self.decoder_input = nn.Linear(latent_dim, fc_input_dim)
        # 3 upsampling layers, 8x8 -> 64x64
        self.decoder = nn.Sequential(
            self._separable_conv(hidden_dims[-1], hidden_dims[-1] // 2, upsample=True),
            self._separable_conv(hidden_dims[-1] // 2, hidden_dims[-1] // 4, upsample=True),
            self._separable_conv(hidden_dims[-1] // 4, hidden_dims[-1] // 8, upsample=True),
        )
        # Final layer: 64x64 -> in_channels
        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1] // 8, self.in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _mobilenetv3_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            # Depthwise Separable Convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(),
            # Pointwise Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )
    
    def _separable_conv(self, in_channels: int, out_channels: int, upsample: bool = False) -> nn.Sequential:
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        # Depthwise Separable Convolution: groups=in_channels
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels))
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.Hardswish())
        # Pointwise Convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.Hardswish())
        return nn.Sequential(*layers)

    def encode(self, encode_input) -> List[Tensor]:
        # Pass the input through the encoder
        res = self.encoder(encode_input)    # [batch_size, hidden_dims[-1], 8, 8]
        # Flatten the output
        res = torch.flatten(res, start_dim=1)    # [batch_size, hidden_dims[-1] * 8 * 8]
        # Compute mean and log variance
        mu = self.fc_mu(res)    # [batch_size, latent_dim]
        logvar = self.fc_logvar(res)    # [batch_size, latent_dim]
        return [mu, logvar]
    
    def decode(self, z: Tensor) -> Tensor:
        # decode the latent variable z
        res = self.decoder_input(z)    # [batch_size, hidden_dims[-1] * 8 * 8]
        encoder_out_size = 8
        res = res.view(-1, self.hidden_dims[-1], encoder_out_size, encoder_out_size)    # [batch_size, hidden_dims[-1], 8, 8]
        result = self.decoder(res)    # upsampling: [batch_size, hidden_dims[-1] // 8, 64, 64]
        result = self.final_layer(result)    # [batch_size, in_channels, 64, 64]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # Reparameterization trick
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        # encode -> sample -> decode, return reconstructed data and latent variables
        mu, logvar = self.encode(data)    # [batch_size, latent_dim]
        z = self.reparameterize(mu, logvar)    # [batch_size, latent_dim]
        recon_x = self.decode(z)    # [batch_size, in_channels, 64, 64]
        return [recon_x, data, mu, logvar, z]

    def loss_function(self, *args, **kwargs) -> dict:
        # Placeholder for loss function logic
        self.num_iterations += 1
        recon_x, data, mu, logvar = args[0], args[1], args[2], args[3]
        kl_weight = kwargs.get('minibatch_weight', 1.0) # minibatch_weight

        recons_loss = F.mse_loss(recon_x, data, reduction='mean')

        # Reconstruction loss
        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, logvar)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]
        else:
            logvar_clamped = torch.clamp(logvar, min=-10, max=10)
            std = torch.exp(0.5 * logvar_clamped)
            gt_dist = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(std)*self.kl_std) # Our goal distribution
            sampled_dist = torch.distributions.normal.Normal(mu, std)
            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist)
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()

        # 🔧 3. β-VAE损失组合
        if self.loss_type == 'B':
            C = torch.clamp(
                self.capacity_max / self.capacity_stop_iteration * self.num_iterations, 
                0, 
                self.capacity_max.item()
            )
            loss = recons_loss + self.gamma * kl_weight * (kl_loss - C).abs()
        else:
            # 标准VAE损失
            loss = recons_loss + self.beta * kl_weight * kl_loss
        
        # 🔧 4. 返回正确的字典格式
        return {
            'VAEloss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kl_loss.detach(),
            'beta': self.beta,
            'gamma': self.gamma,
            'C': C.detach() if self.loss_type == 'B' else torch.tensor(0.0),
            'num_iterations': self.num_iterations,
            'kl_weight': kl_weight
        }

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        # Generate samples from the latent space and return the images
        z = torch.randn(num_samples, self.latent_dim).cuda()
        samples = self.decode(z)
        return samples
    
    def generate(self, data: Tensor, **kwargs) -> Tensor:
        # Generate images from the input image
        return self.forward(data, **kwargs)[0]    # [batch_size, in_channels, 64, 64]
    
    def get_latent(self, data: Tensor, **kwargs) -> Tensor:
        # Get the latent representation of the input image
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        return z    # [batch_size, latent_dim]