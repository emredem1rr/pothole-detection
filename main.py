#!/usr/bin/env python3
"""
Çukur Tespit Projesi - Ana Dosya
"""

import argparse
from src.utils import load_config, create_directories
from src.data_loader import download_dataset, organize_dataset
from src.train import train_model, validate_model
from src.detect import detect_image, detect_video, detect_webcam
import cv2
from pathlib import Path

def setup_project():
    """Projeyi başlangıçta kur"""
    print("="*60)
    print("ÇUKUR TESPİT PROJESİ KURULUMU")
    print("="*60)
    
    config = load_config()
    create_directories(config)
    
    # Veri setini indir
    print("\n1. VERİ SETİ İNDİRİLİYOR...")
    download_dataset(config)
    
    # Veri setini organize et
    print("\n2. VERİ SETİ ORGANİZE EDİLİYOR...")
    organize_dataset(config)
    
    print("\n✓ Proje kurulumu tamamlandı!")
    print("\nŞimdi modeli eğitebilirsiniz:")
    print("  python main.py --train")

def main():
    parser = argparse.ArgumentParser(description='Çukur Tespit Projesi')
    parser.add_argument('--setup', action='store_true', 
                       help='Projeyi kur (veri seti indir ve organize et)')
    parser.add_argument('--train', action='store_true', 
                       help='Modeli eğit')
    parser.add_argument('--validate', action='store_true', 
                       help='Modeli doğrula')
    parser.add_argument('--detect-image', type=str, 
                       help='Görüntüde çukur tespiti yap')
    parser.add_argument('--detect-video', type=str, 
                       help='Video üzerinde çukur tespiti yap')
    parser.add_argument('--webcam', action='store_true', 
                       help='Webcam ile canlı tespit yap')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Konfigürasyon dosyası yolu')
    
    args = parser.parse_args()
    
    # Konfigürasyonu yükle
    config = load_config(args.config)
    
    if args.setup:
        setup_project()
    
    elif args.train:
        train_model(config)
    
    elif args.validate:
        validate_model(config)
    
    elif args.detect_image:
        output_image, results = detect_image(args.detect_image, config)
        
        # Sonucu kaydet
        output_path = Path(config['output']['results_path']) / 'images' / f'output_{Path(args.detect_image).name}'
        cv2.imwrite(str(output_path), output_image)
        print(f"✓ Sonuç kaydedildi: {output_path}")
        
        # Göster
        cv2.imshow('Tespit Sonucu', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.detect_video:
        detect_video(args.detect_video, config)
    
    elif args.webcam:
        detect_webcam(config)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()