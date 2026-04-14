import torch
import torch.nn.functional as F
import torchvision.transforms as T
import random

# Phase 4: Augmentation Transform T(x_hat)
class StyleScriptAugmenter:
    def __init__(self):
        # T_rot: Rotation between -3 and +3 degrees
        self.rotate = T.RandomRotation(degrees=[-3.0, 3.0])
        
        # T_pers: Perspective distortion
        self.perspective = T.RandomPerspective(distortion_scale=0.1, p=1.0)
        
    def apply_noise(self, x, std=0.05):
        # T_noise: Gaussian noise injection η ~ N(0, σ^2)
        noise = torch.randn_like(x) * std
        return x + noise
        
    def apply_photometric(self, x):
        # T_photo: Brightness and contrast U(0.95, 1.05)
        alpha = random.uniform(0.95, 1.05) # contrast
        beta = random.uniform(-0.05, 0.05) # brightness (shifted for normalized tensors)
        return torch.clamp(alpha * x + beta, 0.0, 1.0)

    def forward(self, x):
        # Composite Transform: T = T_rot o T_pers o T_noise o T_photo
        x_aug = self.rotate(x)
        x_aug = self.perspective(x_aug)
        x_aug = self.apply_noise(x_aug)
        x_aug = self.apply_photometric(x_aug)
        return x_aug

# Phase 5: Quality Validation Q(x_aug)
class QualityValidator:
    def __init__(self, blank_threshold=0.98, edge_threshold=0.001):
        self.blank_threshold = blank_threshold
        self.edge_threshold = edge_threshold
        
        # Laplacian kernel to approximate -∇²(G * x) for edge sharpness R
        self.laplacian_kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                                [1.0, -4.0, 1.0],
                                                [0.0, 1.0, 0.0]]]])

    def check_blank(self, x):
        # Q_blank = 1 [min(x) >= tau_white]
        # In a 0.0 to 1.0 tensor, 0.98 approximates the paper's 250/255 threshold
        return torch.min(x) >= self.blank_threshold

    def check_sharpness(self, x):
        # R = -∇²(G * x)
        # Apply laplacian convolution to measure variance (edge sharpness)
        edges = F.conv2d(x, self.laplacian_kernel.to(x.device), padding=1)
        sharpness_score = torch.var(edges)
        return sharpness_score >= self.edge_threshold

    def validate(self, x):
        # Q_total = Q_blank ∧ Q_bound ∧ (R ≥ R_min)
        # Bounding check is implicit as our tensor is strictly constrained to 64x128
        is_blank = self.check_blank(x)
        is_sharp = self.check_sharpness(x)
        
        # Returns 1.0 if valid (Not Blank AND Sharp), else returns a near-zero penalty
        # This will be used in Phase 6 for the -log(Q_total) loss
        if not is_blank and is_sharp:
            return 1.0
        else:
            return 0.1 # Small non-zero value to prevent log(0) errors later

# Quick local test
if __name__ == "__main__":
    # Simulate the output from Phase 3 (1 batch, 1 channel, 64x128)
    dummy_img = torch.rand(1, 1, 64, 128) 
    
    augmenter = StyleScriptAugmenter()
    validator = QualityValidator()
    
    aug_img = augmenter.forward(dummy_img)
    quality_score = validator.validate(aug_img)
    
    print(f"Augmented Image Shape: {aug_img.shape}")
    print(f"Quality Score Output: {quality_score}")