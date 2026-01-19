from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from src.utils import draw_detections
import time

def detect_image(image_path, config, model_path='models/best.pt'):
    """Tek bir görüntüde çukur tespiti yap"""
    # Model yükle
    model = YOLO(model_path)
    
    # Görüntüyü oku
    image = cv2.imread(str(image_path))
    
    # Tespit yap
    results = model(
        image,
        conf=config['detection']['conf_threshold'],
        iou=config['detection']['iou_threshold'],
        max_det=config['detection']['max_detections']
    )
    
    # Tespit sonuçlarını çiz
    output_image = draw_detections(image.copy(), results)
    
    return output_image, results

def detect_video(video_path, config, model_path='models/best.pt'):
    """Video üzerinde çukur tespiti yap"""
    print(f"\nVideo işleniyor: {video_path}")
    
    # Model yükle
    model = YOLO(model_path)
    
    # Video aç
    cap = cv2.VideoCapture(str(video_path))
    
    # Video özellikleri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Çıktı video
    output_path = Path(config['output']['results_path']) / 'videos' / f'output_{Path(video_path).name}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tespit yap
        results = model(
            frame,
            conf=config['detection']['conf_threshold'],
            iou=config['detection']['iou_threshold'],
            verbose=False
        )
        
        # Tespit sonuçlarını çiz
        output_frame = draw_detections(frame, results)
        
        # FPS bilgisi ekle
        fps_text = f'FPS: {frame_count / (time.time() - start_time):.2f}'
        cv2.putText(output_frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Çukur sayısı ekle
        pothole_count = len(results[0].boxes)
        count_text = f'Cukurlar: {pothole_count}'
        cv2.putText(output_frame, count_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(output_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"İşlenen frame: {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    print(f"✓ Video kaydedildi: {output_path}")
    return str(output_path)

def detect_webcam(config, model_path='models/best.pt'):
    """Webcam'den canlı çukur tespiti yap"""
    print("\nWebcam açılıyor... (Çıkmak için 'q' tuşuna basın)")
    
    # Model yükle
    model = YOLO(model_path)
    
    # Webcam aç
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tespit yap
        results = model(
            frame,
            conf=config['detection']['conf_threshold'],
            iou=config['detection']['iou_threshold'],
            verbose=False
        )
        
        # Tespit sonuçlarını çiz
        output_frame = draw_detections(frame, results)
        
        # Çukur sayısı ekle
        pothole_count = len(results[0].boxes)
        count_text = f'Cukurlar: {pothole_count}'
        cv2.putText(output_frame, count_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Cukur Tespiti', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Webcam kapatıldı")