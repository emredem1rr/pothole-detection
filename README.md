# Pothole Detection Project 🚗

Yol çukurlarını gerçek zamanlı olarak tespit etmek için YOLOv8 tabanlı derin öğrenme projesi. Bu proje yol altyapısı inspeksiyonlarını otomatikleştirmek ve kamu güvenliğini artırmak amacıyla geliştirilmiştir.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Teknoloji Stack](#teknoloji-stack)
- [Model İndirme](#model-indirme)
- [Sorun Giderme](#sorun-giderme)
- [Lisans](#lisans)

## ✨ Özellikler

- ✅ **Fotoğrafta Tespit** - Statik görüntülerde çukur tespiti
- ✅ **Videoda Tespit** - Video dosyalarında gerçek zamanlı tespit
- ✅ **Webcam Desteği** - Canlı kameradan görüntü işleme
- ✅ **Model Eğitim** - Kendi veri seti ile model eğitim
- ✅ **YOLOv8 Nano** - Hızlı ve hafif nesne tespit modeli
- ✅ **GPU/CPU Uyumlu** - CUDA desteği ile hızlı işleme
- ✅ **Ayarlanabilir Parametreler** - Confidence ve IoU eşikleri özelleştirilebilir

## 🔧 Gereksinimler

- **Python**: 3.8 veya üzeri
- **pip**: Python paket yöneticisi
- **CUDA** (opsiyonel): GPU hızlandırması için
- **RAM**: Minimum 4GB
- **Disk**: Model dosyası için ~2GB boş alan

## 🚀 Kurulum

### 1. Repository'yi Klonla

```bash
git clone https://github.com/emredem1rr/pothole-detection.git
cd pothole-detection
```

### 2. Virtual Environment Oluştur

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 4. Model Dosyasını Hazırla

```bash
# Otomatik indirme
python main.py --setup
```

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

**Çıkış**: `Q` tuşuna basın

### Modeli Eğit

```bash
python main.py --train
```

### Modeli Doğrula

```bash
python main.py --validate
```

### Projeyi Başlat (Veri Seti İndir)

```bash
python main.py --setup
```

## 📁 Proje Yapısı

```
pothole-detection/
├── main.py                           # Ana giriş noktası
├── requirements.txt                  # Python bağımlılıkları
├── .gitignore                        # Git ignore kuralları
├── README.md                         # Proje dokümantasyonu
│
├── configs/
│   └── config.yaml                  # Proje konfigürasyonu
│
├── data/
│   ├── raw/
│   │   └── dataset/                 # Ham veri seti
│   ├── processed/                   # İşlenmiş veriler
│   └── splits/
│       ├── train/, val/, test/
│
├── src/                              # Kaynak kodlar
│   ├── train.py                     # Eğitim fonksiyonları
│   ├── detect.py                    # Tespit fonksiyonları
│   ├── data_loader.py               # Veri yükleme
│   └── utils.py                     # Yardımcı fonksiyonlar
│
├── test/                             # Test dosyaları
│   ├── test1.jpeg, test2.jpeg       # Test fotoğrafları
│   └── videotest.mp4                # Test videoları
│
├── models/
│   └── best.pt                      # Eğitilmiş YOLOv8 modeli
│
├── results/
│   ├── images/                      # Tespit edilmiş görüntüler
│   └── videos/                      # Tespit edilmiş videolar
│
└── runs/                             # Eğitim sonuçları
```

## 📊 Dataset

Bu projede kullanılan veri seti:
- Açık kaynak yol çukuru (pothole) görüntüleri
- Manuel etiketleme (YOLO formatında)

> Dataset telif ve boyut nedeniyle GitHub reposuna eklenmemiştir.

## 🔧 Teknoloji Stack

| Teknoloji | Kullanım |
|-----------|----------|
| **Python 3.8+** | Programlama dili |
| **YOLOv8** | Nesne tespit modeli |
| **PyTorch** | Deep learning framework |
| **Ultralytics** | YOLO implementasyonu |
| **OpenCV** | Görüntü işleme |
| **NumPy** | Sayısal hesaplamalar |

## 📥 Model İndirme

`best.pt` dosyası GitHub'da yer almaz (dosya boyutu ~2GB). İlk çalıştırmada otomatik indirilir:

```bash
python main.py --setup
```

Veya manuel indirme:

```bash
python -c "from ultralytics import YOLO; YOLO('best.pt')"
```

## 🐛 Sorun Giderme

| Problem | Çözüm |
|---------|-------|
| Model dosyası bulunamadı | `python main.py --setup` çalıştırın |
| GPU tanınmıyor | CUDA sürümünü kontrol edin |
| Düşük performans | GPU kullanın veya model boyutunu azaltın |


## 📧 İletişim

- **GitHub**: [emredem1rr](https://github.com/emredem1rr)
- **Gmail**: demire773@gmail.com


