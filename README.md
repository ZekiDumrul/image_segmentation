# Segmentation Project 

Bu proje, **OpenCV** ve **NumPy** kullanarak farklı **görüntü segmentasyon tekniklerini** karşılaştırmalı olarak uygular.  
Örnek görsel olarak `water_coins.jpg` kullanılmıştır.  

## 🚀 Özellikler

Projede kullanılan segmentasyon yöntemleri:  

- **Otsu Thresholding** → Gri seviye tabanlı ikili eşikleme  
- **Morfolojik İşlemler** → Gürültü temizleme, foreground/background belirleme  
- **Distance Transform** → Öznitelik çıkarımı ve foreground tespiti  
- **GrabCut Algoritması** → Ön/arka plan ayrımı için maskeye dayalı yöntem  
- **Watershed Algoritması** → Nesne ayırma (ör. üst üste gelen paralar)  
- **K-Means Clustering** → Renk tabanlı segmentasyon


## 🔧 Gereksinimler

- Python 3.8+  
- OpenCV  
- NumPy  
- Matplotlib  

Kurulum:  

```bash
pip install opencv-python numpy matplotlib


