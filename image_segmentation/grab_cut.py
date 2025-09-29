import cv2
import numpy as np
import matplotlib.pyplot as plt


# Görüntü yükleme

def load_image(path):
    return cv2.imread(path)


# Ön işleme (gri + Otsu threshold)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, bin_img


# Gürültü giderme ve morfoloji

def remove_noise(bin_img, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    return opened, sure_bg


# Foreground ve unknown alan belirleme

def get_foreground(bin_img_open):
    dist = cv2.distanceTransform(bin_img_open, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(cv2.dilate(bin_img_open, None, iterations=3), sure_fg)
    return dist, sure_fg, unknown


# GrabCut uygulama

def apply_grabcut(img, sure_bg, sure_fg, unknown):
    mask = np.full(img.shape[:2], cv2.GC_PR_BGD, np.uint8)

    mask[sure_bg == 255] = cv2.GC_BGD   # kesin arka plan
    mask[sure_fg == 255] = cv2.GC_FGD   # kesin ön plan
    mask[unknown == 255] = cv2.GC_PR_BGD  # muhtemel arka plan

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    result = img * mask2[:,:,np.newaxis]
    return result, mask2


# Çalıştır

img_path = r"C:\Users\Zeki\Desktop\resim\water_coins.jpg"
img = load_image(img_path)

gray, bin_img = preprocess(img)
opened, sure_bg = remove_noise(bin_img)
dist, sure_fg, unknown = get_foreground(opened)

result, mask2 = apply_grabcut(img, sure_bg, sure_fg, unknown)


# Görselleştir

plt.figure(figsize=(15,6))
plt.subplot(1,3,1); plt.title("Orijinal"); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(1,3,2); plt.title("Binary (Otsu)"); plt.imshow(bin_img, cmap='gray'); plt.axis("off")
plt.subplot(1,3,3); plt.title("GrabCut Sonuç"); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.show()
