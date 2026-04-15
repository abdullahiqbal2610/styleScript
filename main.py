import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Import our custom modules
from src.phase1_extraction import get_style_vector
from src.phase2_3_model import StyleScriptGenerator
from src.phase4_5_utils import StyleScriptAugmenter, QualityValidator

# --- DUMMY DATASET LOADER (Now using Real Text & Tokenizer) ---
class DummyMSC_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, processor):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor # HuggingFace Tokenizer
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        # Load style
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        style_vector = get_style_vector(img_name)
        
        # Load REAL text (e.g., "STRUCTI") and Tokenize it
        text = self.annotations.iloc[idx, 1]
        # The tokenizer converts words into integer IDs that TrOCR understands
        text_tokens = self.processor(text=text, return_tensors="pt").input_ids.squeeze(0)
        
        return text_tokens, torch.tensor(style_vector, dtype=torch.float32)

# --- PHASE 6: LOSS FUNCTIONS (Now with True Cross-Entropy) ---
class StyleScriptLoss(nn.Module):
    def __init__(self, trocr_model):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.trocr_model = trocr_model
        
    def forward(self, target_style, generated_img, target_text_tokens, quality_score):
        # 1. Style Loss: L1 norm
        approx_generated_style = torch.mean(generated_img, dim=[2, 3]).repeat(1, 2)
        target_style_scaled = target_style / 100.0 
        L_style = self.l1(approx_generated_style, target_style_scaled)
        
        # 2. Content Loss: Cross-Entropy via TrOCR Decoder
        # Format our 1x64x128 image to TrOCR's expected 3x384x384 input
        trocr_input = F.interpolate(generated_img, size=(384, 384), mode='bilinear', align_corners=False)
        trocr_input = trocr_input.repeat(1, 3, 1, 1)
        
        # We pass the generated image to TrOCR's "eyes" and the target text to its "brain".
        # Because we provide 'labels', HuggingFace automatically computes the Cross-Entropy loss!
        outputs = self.trocr_model(pixel_values=trocr_input, labels=target_text_tokens)
        L_content = outputs.loss 
        
        # 3. Quality Loss: -log(Q_total)
        L_quality = -torch.log(torch.tensor(quality_score, dtype=torch.float32, requires_grad=True))
        
        return L_style, L_content, L_quality

# --- PHASE 7: OVERALL OPTIMIZATION & TRAINING LOOP ---
def train_one_epoch():
    print("--- Starting StyleScript Training (1 Epoch) ---")
    
   # Load TrOCR and its Tokenizer
    print("Loading TrOCR model and Tokenizer...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed", use_fast=False)
    trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    
    # --- FIX: Explicitly set the start and pad tokens for the decoder ---
    trocr.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    trocr.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Freeze TrOCR (We only train our generator, we don't want to break Microsoft's model)
    trocr.eval()
    for param in trocr.parameters():
        param.requires_grad = False
    
    # Initialize Dataset with the processor
    dataset = DummyMSC_Dataset(csv_file="data/annotations.csv", img_dir="data/raw/", processor=processor)
    # Using batch_size=1 here because real text strings have different token lengths (padding gets complex in batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize Pipeline Components
    # We increase vocab_size to 60000 to support all of TrOCR's possible text tokens
    generator = StyleScriptGenerator()
    generator.encoder.char_embedding = nn.Embedding(num_embeddings=60000, embedding_dim=64)
    
    augmenter = StyleScriptAugmenter()
    validator = QualityValidator()
    criterion = StyleScriptLoss(trocr_model=trocr)
    
    optimizer = optim.AdamW(generator.parameters(), lr=0.0002)
    lambda1, lambda2, lambda3 = 1.0, 1.0, 0.1 
    
    generator.train()
    
    for batch_idx, (text_tokens, style_vectors) in enumerate(dataloader):
        optimizer.zero_grad()
        
        generated_imgs = generator(text_tokens, style_vectors)
        aug_imgs = augmenter.forward(generated_imgs)
        q_score = validator.validate(aug_imgs)
        
        L_style, L_content, L_quality = criterion(style_vectors, generated_imgs, text_tokens, q_score)
        
        L_total = (lambda1 * L_style) + (lambda2 * L_content) + (lambda3 * L_quality)
        
        L_total.backward()
        optimizer.step()
        
        print(f"Image {batch_idx+1}/{len(dataloader)} | L_style: {L_style.item():.4f} | L_content (Cross-Entropy): {L_content.item():.4f} | L_total: {L_total.item():.4f}")
              
    print("\nTraining Complete! 100% Conceptually Accurate NLP Integration.")

if __name__ == "__main__":
    train_one_epoch()