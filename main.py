import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import our custom modules
from src.phase1_extraction import get_style_vector
from src.phase2_3_model import StyleScriptGenerator
from src.phase4_5_utils import StyleScriptAugmenter, QualityValidator

# --- DUMMY DATASET LOADER ---
class DummyMSC_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        text = self.annotations.iloc[idx, 1]
        
        # Phase 1: Extract real style vector using OpenCV
        style_vector = get_style_vector(img_name)
        
        # Convert text to dummy character indices (just for mathematical structural proof)
        text_indices = torch.randint(0, 100, (1, 5)) # Batch 1, 5 random chars
        
        return text_indices.squeeze(0), torch.tensor(style_vector, dtype=torch.float32)

# --- PHASE 6: LOSS FUNCTIONS ---
class StyleScriptLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        
    def forward(self, target_style, generated_img, target_text_indices, quality_score):
        # 1. Style Loss: L1 norm. 
        # generated_img is [Batch, 1, 64, 128]. Mean over H/W gives [Batch, 1].
        # We repeat it to [Batch, 2] to match the target_style [Thickness, Angle].
        approx_generated_style = torch.mean(generated_img, dim=[2, 3]).repeat(1, 2)
        target_style_scaled = target_style / 100.0 
        L_style = self.l1(approx_generated_style, target_style_scaled)
        
        # 2. Content Loss: L2 norm. 
        # Pool across Channels and Height to get [Batch, 128]
        approx_content = torch.mean(generated_img, dim=[1, 2]) 
        # Slice the first N elements to match the text indices shape [Batch, N]
        approx_content = approx_content[:, :target_text_indices.shape[1]] 
        L_content = self.l2(approx_content.float(), target_text_indices.float())
        
        # 3. Quality Loss: -log(Q_total)
        L_quality = -torch.log(torch.tensor(quality_score, dtype=torch.float32, requires_grad=True))
        
        return L_style, L_content, L_quality

# --- PHASE 7: OVERALL OPTIMIZATION & TRAINING LOOP ---
def train_one_epoch():
    print("--- Starting StyleScript Training (1 Epoch) ---")
    
    # Initialize Dataset and DataLoader
    dataset = DummyMSC_Dataset(csv_file="data/annotations.csv", img_dir="data/raw/")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize Pipeline Components
    generator = StyleScriptGenerator()
    augmenter = StyleScriptAugmenter()
    validator = QualityValidator()
    criterion = StyleScriptLoss()
    
    # Optimizer (AdamW as used in the paper)
    optimizer = optim.AdamW(generator.parameters(), lr=0.0002)
    
    # Hyperparameters for final loss balancing
    lambda1, lambda2, lambda3 = 1.0, 1.0, 0.1 
    
    generator.train()
    
    for batch_idx, (text_indices, style_vectors) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 1. Forward Pass (Phases 2 & 3)
        generated_imgs = generator(text_indices, style_vectors)
        
        # 2. Augment & Validate (Phases 4 & 5)
        aug_imgs = augmenter.forward(generated_imgs)
        q_score = validator.validate(aug_imgs)
        
        # 3. Calculate Losses (Phase 6)
        L_style, L_content, L_quality = criterion(style_vectors, generated_imgs, text_indices, q_score)
        
        # 4. Total Optimization Objective (Phase 7)
        L_total = (lambda1 * L_style) + (lambda2 * L_content) + (lambda3 * L_quality)
        
        # Backpropagation
        L_total.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx+1}/{len(dataloader)} | "
              f"L_style: {L_style.item():.4f} | "
              f"L_content: {L_content.item():.4f} | "
              f"L_total: {L_total.item():.4f}")
              
    print("\nTraining Complete! You successfully implemented the math of the paper.")

if __name__ == "__main__":
    train_one_epoch()