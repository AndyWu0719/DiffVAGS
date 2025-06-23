import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from typing import TypeVar, List
from models.gaussian_vae.foundation_models import aux_foundation_model
Tensor = TypeVar('torch.tensor')


class InteractionGate(nn.Module):
    def __init__(self, geo_dim, sem_dim, gate_type="additive"):
        super().__init__()
        self.gate_type = gate_type
        self.geo_dim = geo_dim
        self.sem_dim = sem_dim
        
        if gate_type == "additive":
            self.gate = nn.Sequential(
                nn.Linear(geo_dim + sem_dim, sem_dim),
                nn.Sigmoid()
            )
            self.sem_to_geo_proj = nn.Linear(sem_dim, geo_dim)
            
        elif gate_type == "attention":
            self.geo_attention = nn.MultiheadAttention(geo_dim, num_heads=4, batch_first=True)
            self.sem_attention = nn.MultiheadAttention(sem_dim, num_heads=4, batch_first=True)
    
    def forward(self, geo_feat, sem_feat):
        if self.gate_type == "additive":
            combined = torch.cat([geo_feat, sem_feat], dim=-1)  # [B, geo_dim + sem_dim]
            gate = self.gate(combined)  # [B, sem_dim]
            
            gated_sem = gate * sem_feat  # [B, sem_dim]
            
            sem_proj_to_geo = self.sem_to_geo_proj(gated_sem)  # [B, geo_dim]
            
            enhanced_geo = geo_feat + 0.1 * sem_proj_to_geo
            
            return enhanced_geo, sem_feat
            
        else:
            geo_query = geo_feat.unsqueeze(1)  # [B, 1, geo_dim]
            sem_key_value = sem_feat.unsqueeze(1)  # [B, 1, sem_dim]
            
            if not hasattr(self, 'sem_to_geo_proj_attn'):
                self.sem_to_geo_proj_attn = nn.Linear(sem_feat.shape[-1], geo_feat.shape[-1]).to(geo_feat.device)
            
            sem_proj = self.sem_to_geo_proj_attn(sem_key_value)  # [B, 1, geo_dim]
            
            geo_enhanced, _ = self.geo_attention(geo_query, sem_proj, sem_proj)
            
            return geo_enhanced.squeeze(1), sem_feat

class MultiviewAggregator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, max(64, feature_dim//4)),
            nn.ReLU(),
            nn.Linear(max(64, feature_dim//4), 1)
        )
    
    def forward(self, multiview_features):
        B, N_views, D = multiview_features.shape
        features_flat = multiview_features.view(B * N_views, D)
        attention_scores = self.attention(features_flat).view(B, N_views)
        attention_weights = F.softmax(attention_scores, dim=1)
        return (multiview_features * attention_weights.unsqueeze(-1)).sum(dim=1)

# Gaussian VAE Encoder
class GaussianEncoder(nn.Module):
    num_iterations = 0

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
                 enable_vavae: bool = False,
                 foundation_model_type: str = "dinov2",
                 geo_ratio: float = 0.6,
                 alignment_loss_weight: float = 0.1,
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
        
        self.enable_vavae = enable_vavae
        if self.enable_vavae:
            self.foundation_model_type = foundation_model_type
            self.geo_ratio = geo_ratio
            self.alignment_loss_weight = alignment_loss_weight
            self.distance_scale = nn.Parameter(torch.tensor(1.0))
            self.distance_weight = nn.Parameter(torch.tensor(0.5))
            self.sem_bn = nn.BatchNorm1d(self.foundation_feature_dim)
            self.multiview_aggregator = MultiviewAggregator(
                self.foundation_feature_dim
            )
        
        
        print(f"Initializing GaussianEncoder:")
        print(f"  Input channels: {self.in_channels}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Hidden dims: {self.hidden_dims}")
        print(f"  Enable VAVAE: {self.enable_vavae}")
        print(f"  Loss type: {self.loss_type}")

        self.C_max = torch.Tensor([self.max_capacity])
        self.capacity_max_iteration = self.capacity_max_iteration

        if self.enable_vavae:
            print(f"Initializing VAVAE with {self.foundation_model_type.upper()} foundation model...")
            self.foundation_model = aux_foundation_model(self.foundation_model_type)
            self.foundation_feature_dim = self.foundation_model.feature_dim
            
            self.sem_hidden_multiplier = 2
            self.sem_dropout = 0.1
            self.sem_use_layer_norm = True

            self.interaction_gate = InteractionGate(geo_dim, sem_dim, gate_type="additive")
            self.interaction_strength = nn.Parameter(torch.tensor(0.5))

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
        # VA-VAE specific components
        if self.enable_vavae:
            geo_dim = int(self.latent_dim * self.geo_ratio)
            sem_dim = self.latent_dim - geo_dim
            
            self.fc_mu_geo = nn.Linear(feature_dim, geo_dim)
            self.fc_logvar_geo = nn.Linear(feature_dim, geo_dim)
            
            self.fc_mu_sem = nn.Linear(feature_dim, sem_dim)
            self.fc_logvar_sem = nn.Linear(feature_dim, sem_dim)
            
            self.semantic_projector = nn.Sequential(
                nn.Linear(sem_dim, sem_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(sem_dim * 2, self.foundation_feature_dim),
                nn.LayerNorm(self.foundation_feature_dim)
            )
            
            self.geo_dim = geo_dim
            self.sem_dim = sem_dim
            
            print(f"VAVAE latent space: geo_dim={geo_dim}, sem_dim={sem_dim}")
            print(f"Foundation model feature dim: {self.foundation_feature_dim}")
        else:
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
        
        # Compute mu and logvar
        if self.enable_vavae:
            mu_geo = self.fc_mu_geo(result)
            logvar_geo = self.fc_logvar_geo(result)
            mu_sem = self.fc_mu_sem(result) 
            logvar_sem = self.fc_logvar_sem(result)
            
            return [mu_geo, logvar_geo, mu_sem, logvar_sem]
        else:
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
    
    def extract_multiview_features(self, multiview_images: Tensor) -> Tensor:

        B, N_views, C, H, W = multiview_images.shape
        
        images_flat = multiview_images.view(B * N_views, C, H, W)  # [B*N_views, C, H, W]
        
        with torch.no_grad():
            global_features_flat = self.foundation_model.extract_global_features(images_flat)  # [B*N_views, feature_dim]
        
        return global_features_flat.view(B, N_views, self.foundation_feature_dim)
    
    def forward(self, data: Tensor, multiview_images: Tensor = None, **kwargs) -> Tensor:
        if self.enable_vavae:
            if multiview_images is None:
                raise ValueError("VAVAE mode requires multiview_images for visual alignment!")
            
            mu_geo, logvar_geo, mu_sem, logvar_sem = self.encode(data)
            
            z_geo = self.reparameterize(mu_geo, logvar_geo)
            z_sem = self.reparameterize(mu_sem, logvar_sem)

            if hasattr(self, 'interaction_gate'):
                z_geo_enhanced, z_sem_enhanced = self.interaction_gate(z_geo, z_sem)
                alpha = torch.sigmoid(self.interaction_strength)
                z_geo = (1 - alpha) * z_geo + alpha * z_geo_enhanced
                z_sem = (1 - alpha) * z_sem + alpha * z_sem_enhanced
            
            z = torch.cat([z_geo, z_sem], dim=-1)
            
            recon_x = self.decode(z)

            multiview_features = self.extract_multiview_features(multiview_images)
            
            alignment_loss, pointwise_loss, distance_loss = self.compute_visual_alignment_loss(z_sem, multiview_features)
            
            return [recon_x, data, mu_geo, logvar_geo, mu_sem, logvar_sem, z_geo, z_sem, alignment_loss, pointwise_loss, distance_loss]
        else:
            mu, logvar = self.encode(data)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
            
            return [recon_x, data, mu, logvar, z]
        
    
    def _compute_distance_matrix_loss(self, Z, V, temperature):
        if Z.size(0) > 64:
            indices = torch.randperm(Z.size(0))[:64]
            Z, V = Z[indices], V[indices]
        
        Z_sim = F.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=-1)
        V_sim = F.cosine_similarity(V.unsqueeze(1), V.unsqueeze(0), dim=-1)
        
        scale = getattr(self, 'distance_scale', 1.0)
        Z_sim = scale * Z_sim / temperature
        V_sim = scale * V_sim / temperature
        
        return F.mse_loss(Z_sim, V_sim)
        
    def compute_visual_alignment_loss(self, z_sem: Tensor, multiview_features: Tensor, margin: float = 0.9, temperature: float = 1.0, distance_weight: float = 0.5) -> Tensor:

        sem_projected = self.semantic_projector(z_sem)  # [B, foundation_feature_dim]
        sem_projected = self.sem_bn(sem_projected)
        Z = F.normalize(sem_projected, p=2, dim=-1)  # [B, foundation_feature_dim]
        
        V = self.multiview_aggregator(multiview_features)
        V = F.normalize(V, p=2, dim=-1)  # [B, foundation_feature_dim]
        
        cosine_sims = torch.sum(Z * V, dim=-1)  # [B]
        pointwise_loss = F.relu(margin - cosine_sims).mean()
        
        if Z.shape[0] > 1:
            distance_loss = self._compute_distance_matrix_loss(Z, V, temperature)
        else:
            distance_loss = torch.tensor(0.0, device=Z.device)
        
        total_alignment_loss = pointwise_loss + distance_weight * distance_loss
            
        if hasattr(self, '_loss_stats'):
            self._loss_stats = {
                'pointwise_loss': pointwise_loss.item(),
                'distance_loss': distance_loss.item(),
                'total_alignment_loss': total_alignment_loss.item()
            }
        
        if temperature != 1.0:
            total_alignment_loss = total_alignment_loss / temperature
        
        return total_alignment_loss, pointwise_loss, distance_loss
    
    def loss_function(self, *args, **kwargs) -> dict:
        # Placeholder for loss function logic
        self.num_iterations += 1
        
        kld_weight = kwargs.get('M_N', 1.0)  # minibatch weight
        
        if self.enable_vavae:
            recons, data, mu_geo, logvar_geo, mu_sem, logvar_sem, z_geo, z_sem, alignment_loss, pointwise_loss, distance_loss = args
            
            recons_loss = F.mse_loss(recons, data, reduction='mean')
            
            kl_geo = self._compute_kl_loss(mu_geo, logvar_geo)
            kl_sem = self._compute_kl_loss(mu_sem, logvar_sem)
            kl_loss = kl_geo + kl_sem
            
            if self.loss_type == 'B':
                device = recons.device
                C_max = self.C_max.to(device)

                C = torch.clamp(
                    C_max / self.capacity_max_iteration * GaussianEncoder.num_iterations,
                    0, C_max.item()
                )
                base_vae_loss = recons_loss + self.gamma * kld_weight * (kl_loss - C).abs()
            else:
                base_vae_loss = recons_loss + self.beta * kld_weight * kl_loss
                C = torch.tensor(0.0, device=recons.device)
            
            vavae_loss = base_vae_loss + self.alignment_loss_weight * alignment_loss
            
            return {
                'VAEloss': vavae_loss,
                'Reconstruction_Loss': recons_loss.detach(),
                'KLD_geo': kl_geo.detach(),
                'KLD_sem': kl_sem.detach(), 
                'KLD_total': kl_loss.detach(),
                'VF_Loss': alignment_loss.detach(),
                'VF_Pointwise_Loss': pointwise_loss.detach(),
                'VF_Distance_Loss': distance_loss.detach(),
                'Base_VAE_Loss': base_vae_loss.detach(),
                'Interaction_Strength': torch.sigmoid(self.interaction_strength).detach() if hasattr(self, 'interaction_strength') else 0.5,
                'beta': self.beta,
                'gamma': self.gamma,
                'C': C.detach(),
                'num_iterations': self.num_iterations
            }
            
        else:
            recons, data, mu, logvar, z = args
            kl_loss = self._compute_kl_loss(mu, logvar)
            vae_loss = kld_weight * kl_loss
            
            return {
                'VAEloss': vae_loss
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
        if self.enable_vavae:
            z_geo = self.get_geometric_latent(self, x)
            z_sem = self.get_semantic_latent(self, x)
            z = torch.cat([z_geo, z_sem], dim=-1)
            return z
        else:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return z
    
    def get_semantic_latent(self, x: Tensor) -> Tensor:
        mu_geo, logvar_geo, mu_sem, logvar_sem = self.encode(x)
        z_sem = self.reparameterize(mu_sem, logvar_sem)
        return z_sem
    
    def get_geometric_latent(self, x: Tensor) -> Tensor:
        mu_geo, logvar_geo, mu_sem, logvar_sem = self.encode(x)
        z_geo = self.reparameterize(mu_geo, logvar_geo)
        return z_geo