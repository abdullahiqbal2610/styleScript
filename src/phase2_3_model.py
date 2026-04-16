import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Phase 2: Character-Level Text Encoder E(y)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=64):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, text_indices):
        # 1. Character Embedding: E(y) = [fc1, fc2, ..., fcL]  (Eq. 4)
        embeddings = self.char_embedding(text_indices)
        
        # 2. Noise Modulation: F(y, ε) = fc ⊗ ε  (Eq. 5)
        noise = torch.randn_like(embeddings)
        modulated = embeddings * noise
        
        # 3. Text Map: M = concat_horiz(F)  (Eq. 6)
        M = modulated.permute(0, 2, 1).unsqueeze(2) 
        return M

# Phase 3: Style-Controlled Generator G(M, s)
class ConditionalBatchNorm2d(nn.Module):
    """ Implements CBN(h, s) = γ(s) * ((h - μ)/σ) + β(s)  (Eq. 7) """
    def __init__(self, num_features, style_dim=2):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_fc = nn.Linear(style_dim, num_features)
        self.beta_fc = nn.Linear(style_dim, num_features)

    def forward(self, x, style_vector):
        out = self.bn(x)
        gamma = self.gamma_fc(style_vector).view(-1, x.size(1), 1, 1)
        beta = self.beta_fc(style_vector).view(-1, x.size(1), 1, 1)
        return gamma * out + beta

class StyleScriptGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TextEncoder()
        self.conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.cbn = ConditionalBatchNorm2d(1, style_dim=2)
        
    def forward(self, text_indices, style_vector):
        # Phase 2: Get Text Map M
        M = self.encoder(text_indices)
        
        # Phase 3: Apply Convolutions and Conditional Batch Norm
        out = self.conv(M)
        out = self.cbn(out, style_vector)
        
        # --- NEW: Eq. 8 & Eq. 9 (Font Scaling and Shear Transformation) ---
        batch_size = out.size(0)
        tau = style_vector[:, 0]        # Stroke thickness
        theta_deg = style_vector[:, 1]  # Slant angle
       
        # Eq. 8: Font Size Adj Factor = min(max(tau/2, 0.8), 1.2)
        font_scale = torch.clamp(tau / 2.0, min=0.8, max=1.2)
        
        # Eq. 9: Shear Trans Matrix. Clamp between -30 and 30 degrees.
        theta_clamped = torch.clamp(theta_deg, min=-30.0, max=30.0)
        theta_rad = theta_clamped * (math.pi / 180.0)
        shear_factor = torch.tan(theta_rad)
        
        # Create Affine Matrices [Batch, 2, 3] for PyTorch grid mapping
        affine_matrices = torch.zeros(batch_size, 2, 3, device=out.device)
        for i in range(batch_size):
            # PyTorch affine mapping is target-to-source, so we use 1/scale
            s = 1.0 / font_scale[i] 
            sh = shear_factor[i]
            
            # [ sx,  sh, 0 ]
            # [ 0,   sy, 0 ]
            affine_matrices[i, 0, 0] = s
            affine_matrices[i, 0, 1] = sh
            affine_matrices[i, 1, 1] = s

        # Apply the geometric math transformation
        grid = F.affine_grid(affine_matrices, out.size(), align_corners=False)
        out = F.grid_sample(out, grid, align_corners=False, padding_mode='zeros')
        
        # Force output to 64x128 as specified in the paper — x̂ = G(M, s) ∈ ℝ^(H×W_var)  (Eq. 10)
        out = F.interpolate(out, size=(64, 128), mode='bilinear', align_corners=False)
        return out

# Quick local test
if __name__ == "__main__":
    generator = StyleScriptGenerator()
    dummy_text = torch.tensor([[10, 25, 3, 44, 12]]) 
    dummy_style = torch.tensor([[1.0, 15.0]], dtype=torch.float32) # Scale 1.0, Shear 15 deg
    generated_image = generator(dummy_text, dummy_style)
    print(f"Success! Scaled & Sheared Image Tensor Shape: {generated_image.shape}")