from ultralytics import YOLO

# Path ke data.yaml (sesuaikan!)
data_yaml = "C:\PROJECT\PY\ppe-detection\data\datasets\PPE Detection.v3-allclasses\data.yaml"

print("📊 Checking dataset...")
print("=" * 50)

# Load dataset info
from ultralytics.data import YOLODataset
import yaml

with open(data_yaml, 'r') as f:
    data = yaml.safe_load(f)

print(f"✅ Number of classes: {data['nc']}")
print(f"✅ Class names: {data['names']}")
print(f"✅ Train path: {data['train']}")
print(f"✅ Valid path: {data['val']}")
print("\n🎉 Dataset configuration looks good!")