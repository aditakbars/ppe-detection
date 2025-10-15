from ultralytics import YOLO
import torch

print("🔍 Checking device...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")
print("=" * 50)

# Load pretrained model (YOLOv11 nano - paling ringan)
print("📥 Loading YOLOv11 nano model...")
model = YOLO('yolo11n.pt')
print("✅ Model loaded!")

# Path ke data.yaml
data_yaml = "C:\PROJECT\PY\ppe-detection\data\datasets\PPE Detection.v3-allclasses\data.yaml"

print("\n🎯 Starting training...")
print("⏳ Ini bakal lama (bisa 4-8 jam), santai aja ya!")
print("💡 Tips: Jangan ditutup laptop-nya, colok charger!")
print("=" * 50)

# Training dengan config untuk laptop pas-pasan
results = model.train(
    data=data_yaml,
    epochs=50,              # Jumlah iterasi (50 dulu, nanti bisa ditambah)
    imgsz=416,              # Ukuran image (416 lebih ringan dari 640)
    batch=4,                # Batch size kecil (4 atau 8, sesuaikan laptop)
    device=device,          # CPU atau GPU
    patience=10,            # Stop kalau 10 epoch gak ada improvement
    save=True,              # Save model tiap epoch
    project='models',       # Folder output
    name='ppe_model_v1',    # Nama run
    exist_ok=True,          # Overwrite kalau udah ada
    plots=True,             # Bikin grafik hasil training
    verbose=True            # Tampilkan detail progress
)

print("\n🎉 Training selesai!")
print(f"📁 Model tersimpan di: models/ppe_model_v1/weights/best.pt")