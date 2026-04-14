import os
import cv2
import numpy as np
import csv

# Setup directories
os.makedirs("data/raw", exist_ok=True)

dummy_texts = ["STRUCTI", "VESSEL", "WELDING", "SCALE", "NAVSEA", "MARINE", "MATER", "SHIP", "REAR", "DRAW"]

with open("data/annotations.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "text"])
    
    for i, text in enumerate(dummy_texts):
        # Create a 64x128 blank grayscale image (HxW)
        img = np.ones((64, 128), dtype=np.uint8) * 255
        
        # Add text to simulate a cropped word
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (10, 40), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Add a random line to simulate the noisy 'engineering document' feel
        cv2.line(img, (0, np.random.randint(10, 50)), (128, np.random.randint(10, 50)), (150, 150, 150), 1)
        
        filename = f"dummy_{i}.png"
        cv2.imwrite(f"data/raw/{filename}", img)
        writer.writerow([filename, text])

print("Dummy dataset created in data/raw/ with annotations in data/annotations.csv")