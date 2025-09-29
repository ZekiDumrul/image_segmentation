import cv2
import numpy as np
import matplotlib.pyplot as plt


# Görüntü gösterme fonksiyonu

def imshow(img, ax, cmap=None, title=None):
    ax.imshow(img if cmap is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    if title:
        ax.set_title(title)


# Görüntü yükleme

def load_image(path):
    return cv2.imread(path)


# Ön işleme (griye çevirme ve threshold)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, bin_img


# Gürültü giderme ve morfolojik işlemler

def remove_noise(bin_img, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    return opened, sure_bg


# Foreground ve unknown alan belirleme

def get_foreground(bin_img_open):
    dist = cv2.distanceTransform(bin_img_open, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(cv2.dilate(bin_img_open, None, iterations=3), sure_fg)
    return dist, sure_fg, unknown


# Marker oluşturma

def get_markers(sure_fg, unknown):
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    return markers


# Watershed uygulama

def apply_watershed(img, markers):
    markers_ws = cv2.watershed(img.copy(), markers)
    return markers_ws


# Kontur çıkarma

def extract_contours(markers_ws, img):
    labels = np.unique(markers_ws)
    coins = []
    for label in labels[2:]:
        target = np.where(markers_ws == label, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coins.append(contours[0])
    img_contours = img.copy()
    cv2.drawContours(img_contours, coins, -1, color=(0, 23, 223), thickness=2)
    return img_contours


# Marker ve watershed etiketlerini renkli gösterme

def label2color(label_img):
    label_hue = np.uint8(179 * label_img / np.max(label_img))
    blank_ch = 255 * np.ones_like(label_hue)
    colored_img = cv2.merge([label_hue, blank_ch, blank_ch])
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_HSV2RGB)
    return colored_img


# Görselleştirme

def plot_results(images, cols=4, figsize=(16, 12)):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for ax, (image, title) in zip(axes, images):
        if title in ["Gray", "Threshold", "Noise Removed", "Sure Background",
                     "Sure Foreground", "Unknown"]:
            imshow(image, ax, cmap="gray", title=title)
        elif title in ["Markers", "Watershed"]:
            imshow(label2color(image), ax, title=title)
        else:
            imshow(image, ax, title=title)

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Ana fonksiyon

def main(image_path):
    img = load_image(image_path)
    gray, bin_img = preprocess(img)
    bin_img_open, sure_bg = remove_noise(bin_img)
    dist, sure_fg, unknown = get_foreground(bin_img_open)
    markers = get_markers(sure_fg, unknown)
    markers_ws = apply_watershed(img, markers)
    img_contours = extract_contours(markers_ws, img)

    images = [
        (img, "Original"),
        (gray, "Gray"),
        (bin_img, "Threshold"),
        (bin_img_open, "Noise Removed"),
        (sure_bg, "Sure Background"),
        (dist, "Distance Transform"),
        (sure_fg, "Sure Foreground"),
        (unknown, "Unknown"),
        (markers, "Markers"),
        (markers_ws, "Watershed"),
        (img_contours, "Contours"),
    ]

    plot_results(images)


# Çalıştır

if __name__ == "__main__":
    image_path = r"C:\Users\Zeki\Desktop\resim\water_coins.jpg"
    main(image_path)
