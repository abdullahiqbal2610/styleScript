import torch
import jiwer
import pandas as pd
import json
import sys
from pathlib import Path
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch.optim as optim
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase2_3_model import StyleScriptGenerator
from src.phase1_extraction import get_style_vector
import os
import cv2

# Load Config
with open('config.json', 'r') as f:
    config = json.load(f)

print("Loading Models for Downstream Task...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed", use_fast=False)
trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
trocr.config.decoder_start_token_id = processor.tokenizer.cls_token_id
trocr.config.pad_token_id = processor.tokenizer.pad_token_id

# Load the trained StyleScript Generator
generator = StyleScriptGenerator()
generator.encoder.char_embedding = torch.nn.Embedding(num_embeddings=config['hyperparameters']['vocab_size'], embedding_dim=64)
# FIX: Removed the "../" so it looks in the current folder
generator.load_state_dict(torch.load(config['paths']['model_save_path']))
generator.eval()

# FIX: Removed the "../" 
annotations = pd.read_csv(config['paths']['annotations_csv'])

def calculate_metrics(model, annotations_df):
    """ Evaluates the model to find Character Error Rate (CER) and Word Error Rate (WER) """
    model.eval()
    predictions = []
    references = []
    
    for idx, row in annotations_df.iterrows():
        # FIX: Removed the "../"
        img_path = os.path.join(config['paths']['img_dir'], row['filename'])
        true_text = str(row['text'])
        
        # Format image for TrOCR
        img = cv2.imread(img_path)
        if img is None: continue
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        
        # Generate text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        predictions.append(generated_text)
        references.append(true_text)
        
    cer = jiwer.cer(references, predictions)
    wer = jiwer.wer(references, predictions)
    return cer, wer

# 1. BASELINE EVALUATION
print("\n--- Running Baseline Evaluation (Standard TrOCR) ---")
base_cer, base_wer = calculate_metrics(trocr, annotations)
print(f"Baseline CER: {base_cer:.4f} | Baseline WER: {base_wer:.4f}")

# 2. SYNTHETIC GENERATION (Phase 8)
print("\n--- Generating Synthetic Data (Phase 8) ---")
synthetic_data = []
for idx, row in annotations.iterrows():
    true_text = str(row['text'])
    # FIX: Removed the "../"
    img_path = os.path.join(config['paths']['img_dir'], row['filename'])
    style_vec = get_style_vector(img_path)
    
    # Perturb the style vector slightly to create synthetic variation
    synthetic_style = torch.tensor([[style_vec[0] * 1.1, style_vec[1] + 5.0]], dtype=torch.float32)
    text_tokens = processor(text=true_text, return_tensors="pt").input_ids
    
    with torch.no_grad():
        syn_img = generator(text_tokens, synthetic_style)
        
    # Resize tensor to match TrOCR inputs and append to training list
    syn_img_resized = F.interpolate(syn_img, size=(384, 384), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)
    synthetic_data.append((syn_img_resized, text_tokens.squeeze(0)))

# 3. DOWNSTREAM FINE-TUNING (Phase 9)
print("\n--- Fine-Tuning Downstream OCR with StyleScript Data (Phase 9) ---")
trocr.train()
optimizer = optim.AdamW(trocr.parameters(), lr=0.00005) # Tiny learning rate for fine-tuning

for epoch in range(1): # Just 1 epoch to prove the pipeline works
    for syn_img, text_tokens in synthetic_data:
        optimizer.zero_grad()
        outputs = trocr(pixel_values=syn_img, labels=text_tokens.unsqueeze(0))
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 4. STYLESCRIPT EVALUATION
print("\n--- Running StyleScript Evaluation ---")
style_cer, style_wer = calculate_metrics(trocr, annotations)

# 5. FINAL TABLE (Just like the paper!)
print("\n" + "="*60)
print(" 📊 TABLE 1: DOWNSTREAM OCR PERFORMANCE COMPARISON")
print("="*60)
print(f"{'Model':<25} | {'CER (Lower is better)':<20} | {'WER'}")
print("-" * 60)
print(f"{'Baseline TrOCR':<25} | {base_cer:<20.4f} | {base_wer:.4f}")
print(f"{'StyleScript Enhanced OCR':<25} | {style_cer:<20.4f} | {style_wer:.4f}")
print("="*60)
