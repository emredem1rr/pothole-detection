import os
import yaml
import cv2
import numpy as np
from pathlib import Path
import shutil

def load_config(config_path="configs/config.yaml"):
    """Konfigürasyon dosyasını yükle"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(config):
    """Gerekli dizinleri oluştur"""
    paths = [
        config['data']['raw_path'],
        config['data']['processed_path'],
        config['data']['splits_path'],
        os.path.join(config['data']['splits_path'], 'train', 'images'),
        os.path.join(config['data']['splits_path'], 'train', 'labels'),
        os.path.join(config['data']['splits_path'], 'val', 'images'),
        os.path.join(config['data']['splits_path'], 'val', 'labels'),
        os.path.join(config['data']['splits_path'], 'test', 'images'),
        os.path.join(config['data']['splits_path'], 'test', 'labels'),
        'models',
        os.path.join(config['output']['results_path'], 'images'),
        os.path.join(config['output']['results_path'], 'videos'),
    ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)
    
    print("✓ Tüm dizinler oluşturuldu")

def draw_detections(image, results, class_names=['pothole']):
    """Tespit edilen çukurları görüntü üzerine çiz"""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Koordinatları al
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Dikdörtgen çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Etiket oluştur
            label = f'{class_names[cls]}: {conf:.2f}'
            
            # Etiket arka planı
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
            
            # Etiket metni
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

def convert_yolo_to_xyxy(yolo_bbox, img_width, img_height):
    """YOLO formatını (x_center, y_center, w, h) -> (x1, y1, x2, y2) dönüştür"""
    x_center, y_center, w, h = yolo_bbox
    
    x1 = int((x_center - w/2) * img_width)
    y1 = int((y_center - h/2) * img_height)
    x2 = int((x_center + w/2) * img_width)
    y2 = int((y_center + h/2) * img_height)
    
    return [x1, y1, x2, y2]