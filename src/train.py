from ultralytics import YOLO
import torch
from pathlib import Path

def train_model(config):
    """Modeli eğit"""
    print("\n" + "="*50)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("="*50 + "\n")
    
    # Device ayarla
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA bulunamadı, CPU kullanılacak")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    # Model yükle
    model_name = config['model']['name']
    print(f"Model yükleniyor: {model_name}")
    model = YOLO(f'{model_name}.pt')
    
    # Data.yaml yolunu al
    data_yaml = Path(config['data']['splits_path']) / 'data.yaml'
    
    # Eğitim parametreleri
    results = model.train(
        data=str(data_yaml),
        epochs=config['training']['epochs'],
        imgsz=config['model']['img_size'],
        batch=config['training']['batch_size'],
        lr0=config['training']['learning_rate'],
        patience=config['training']['patience'],
        device=device,
        workers=config['training']['workers'],
        project='runs/train',
        name='pothole_detection',
        exist_ok=True,
        pretrained=config['model']['pretrained'],
        verbose=True
    )
    
    # En iyi modeli kaydet
    best_model_path = 'runs/train/pothole_detection/weights/best.pt'
    import shutil
    shutil.copy(best_model_path, 'models/best.pt')
    
    print("\n✓ Model eğitimi tamamlandı!")
    print(f"✓ En iyi model kaydedildi: models/best.pt")
    
    return results

def validate_model(config):
    """Modeli doğrula"""
    print("\nModel doğrulanıyor...")
    
    model = YOLO('models/best.pt')
    data_yaml = Path(config['data']['splits_path']) / 'data.yaml'
    
    results = model.val(
        data=str(data_yaml),
        imgsz=config['model']['img_size'],
        batch=config['training']['batch_size'],
        device=config['training']['device']
    )
    
    print("✓ Doğrulama tamamlandı")
    return results