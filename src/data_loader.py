import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import yaml

def download_dataset(config):
    """Kaggle'dan veri setini indir"""
    print("Veri seti indiriliyor...")
    
    # Veri setini indir
    path = kagglehub.dataset_download(config['data']['kaggle_dataset'])
    
    print(f"✓ Veri seti indirildi: {path}")
    
    # Raw klasörüne kopyala
    raw_path = config['data']['raw_path']
    if os.path.exists(raw_path):
        shutil.rmtree(raw_path)
    
    shutil.copytree(path, raw_path)
    print(f"✓ Veri {raw_path} klasörüne kopyalandı")
    
    return raw_path

def organize_dataset(config):
    """Veri setini organize et ve train/val/test olarak ayır"""
    print("\nVeri seti organize ediliyor...")
    
    raw_path = Path(config['data']['raw_path'])
    splits_path = Path(config['data']['splits_path'])
    
    # Tüm görüntü ve etiket dosyalarını bul
    images = list(raw_path.rglob('*.jpg')) + list(raw_path.rglob('*.png'))
    
    print(f"Toplam {len(images)} görüntü bulundu")
    
    # Karıştır
    random.shuffle(images)
    
    # Bölümlere ayır
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    train_idx = int(len(images) * train_ratio)
    val_idx = int(len(images) * (train_ratio + val_ratio))
    
    train_images = images[:train_idx]
    val_images = images[train_idx:val_idx]
    test_images = images[val_idx:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Her bölüm için dosyaları kopyala
    for split_name, split_images in [('train', train_images), 
                                      ('val', val_images), 
                                      ('test', test_images)]:
        
        img_dir = splits_path / split_name / 'images'
        lbl_dir = splits_path / split_name / 'labels'
        
        for img_path in tqdm(split_images, desc=f"{split_name} kopyalanıyor"):
            # Görüntüyü kopyala
            shutil.copy(img_path, img_dir / img_path.name)
            
            # Etiket dosyasını bul ve kopyala
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy(label_path, lbl_dir / label_path.name)
    
    print("✓ Veri seti organize edildi")
    
    # YOLOv8 için data.yaml oluştur
    create_data_yaml(config)

def create_data_yaml(config):
    """YOLOv8 için data.yaml dosyası oluştur"""
    splits_path = Path(config['data']['splits_path']).resolve()
    
    data_yaml = {
        'path': str(splits_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'pothole'
        },
        'nc': 1  # number of classes
    }
    
    yaml_path = splits_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✓ data.yaml oluşturuldu: {yaml_path}")
    
    return yaml_path