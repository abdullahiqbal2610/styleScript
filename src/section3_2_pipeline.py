import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

def run_trocr_inference(image_path):
    print(f"--- Running Section 3.2: Practical OCR Pipeline ---")
    print(f"Loading TrOCR model (this may take a minute to download weights)...")
    
    # 1. Load the processor and model from Hugging Face
    # We use 'small-printed' as directly benchmarked in the paper
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed", use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    
    # 2. Open the image we generated in Phase 1
    image = Image.open(image_path).convert("RGB")
    
    # 3. Process the image into pixel values
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # 4. Generate the predicted text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Target Image: {image_path}")
    print(f"Recognized Text: '{generated_text}'")
    print("---------------------------------------------------")
    
    return generated_text

if __name__ == "__main__":
    # Test the OCR on the first dummy image we created earlier
    run_trocr_inference("data/raw/dummy_0.png")