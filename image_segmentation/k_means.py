import cv2
import numpy as np
import matplotlib.pyplot as plt

# gorsel yukle
img_path = r"C:\Users\Zeki\Desktop\resim\water_coins.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# image 2D arraye donustur
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# K-means kriterleri ve kume sayisi
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2  # 2 sýnýf (on plan / arka plan)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Merkezleri uint8 yap
centers = np.uint8(centers)

# Piksel degerlerini yeni renklere ata
segmented_img = centers[labels.flatten()]
segmented_img = segmented_img.reshape(img.shape)

# Sonuclarý goster
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Orijinal Gorsel")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmented_img)
plt.title("K-Means Segmentasyonu")
plt.axis("off")

plt.show()

