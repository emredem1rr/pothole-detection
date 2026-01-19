# Pothole Detection Project 🚗

Yol çukurlarını gerçek zamanlı olarak tespit etmek için YOLOv8 tabanlı derin öğrenme projesi.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red)

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)

## ✨ Özellikler

- ✅ Fotoğraflarda çukur tespiti
- ✅ Videolarda çukur tespiti
- ✅ Webcam ile canlı tespit
- ✅ Model eğitim ve doğrulama
- ✅ YOLOv8 nano modeli (hızlı ve hafif)
- ✅ GPU/CPU desteği

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- pip

### Adımlar

1. **Repository klonla**
```bash
git clone https://github.com/[username]/pothole-detection.git
cd pothole-detection
```

2. **Virtual environment oluştur**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Bağımlılıkları yükle**
```bash
pip install -r requirements.txt
```

4. **Modeli indir** (ilk çalıştırmada otomatik indirilir)
```bash
python main.py --setup
```
Veya manuel olarak best.pt indir: [Release sayfasından indir](https://github.com/[username]/pothole-detection/releases)

## 💻 Kullanım

### Fotoğrafta Tespit
```bash
python main.py --detect-image "test/test1.jpeg"
```

### Videoda Tespit
```bash
python main.py --detect-video "test/videotest.mp4"
```

### Webcam ile Canlı Tespit
```bash
python main.py --webcam
```

### Modeli Eğit
```bash
python main.py --train
```

### Modeli Doğrula
```bash
python main.py --validate
```

### Projeyi Kur
```bash
python main.py --setup
```

## 📁 Proje Yapısı

```
pothole-detection/
├── main.py                 # Ana giriş noktası
├── requirements.txt        # Python bağımlılıkları
├── configs/
│   └── config.yaml        # Konfigürasyon dosyası
├── data/
│   ├── raw/               # Ham veri seti
│   ├── processed/         # İşlenmiş veriler
│   └── splits/            # Train/val/test bölümleri
├── src/
│   ├── __init__.py
│   ├── train.py           # Eğitim fonksiyonları
│   ├── detect.py          # Tespit fonksiyonları
│   ├── data_loader.py     # Veri yükleme
│   └── utils.py           # Yardımcı fonksiyonlar
├── test/                  # Test fotoğraf ve videolar
│   ├── test1.jpeg
│   ├── test2.jpeg
│   ├── videotest.mp4
│   └── ...
├── models/
│   └── best.pt            # Eğitilmiş model
└── results/               # Tespit sonuçları
    ├── images/            # Tespit edilmiş görüntüler
    └── videos/            # Tespit edilmiş videolar
```

## 🔧 Teknolojiler

- **Framework**: PyTorch + Ultralytics YOLO
- **Görüntü İşleme**: OpenCV
- **Veri İşleme**: Pandas, NumPy
- **Model**: YOLOv8 Nano

## � Model İndirme

`best.pt` dosyası GitHub'da yer almaz (dosya çok büyük). İlk çalıştırmada otomatik indirilir veya:

```bash
# Manuel indirme
python -c "from ultralytics import YOLO; YOLO('best.pt')"
```

Veya [Release](https://github.com/[username]/pothole-detection/releases) sayfasından indir.

## �📝 Lisans

MIT Lisansı

## 📧 İletişim

Sorular ve öneriler için iletişime geçebilirsiniz.
