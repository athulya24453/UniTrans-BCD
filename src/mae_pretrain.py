import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTMAEForPreTraining, ViTImageProcessor
from sysu_mae_dataset import SysuMAEDataset

# === Configuration ===
DATA_DIR = "/storage2/ChangeDetection/datasets/sysu_preprocessed/t1/train"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "models/mae_pretrained/encoder.pth"

# === Dataset & Dataloader ===
dataset = SysuMAEDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# === Load Hugging Face MAE model ===
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
model.to(DEVICE)

# === Optimizer & Loss ===
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# === Training Loop ===
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in dataloader:
        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)

        outputs = model(**inputs)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# === Save Encoder Only ===
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.vit.state_dict(), SAVE_PATH)
print(f"✅ MAE encoder saved to {SAVE_PATH}")
