import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Sabit değerler
IMAGE_SIZE = (128, 128) 
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 10
TRAIN_TEST_SPLIT = 0.3
MAX_IMAGES_PER_CLASS = 650

# Veri yolu ayarlari
DATA_PATH = r"C:\Users\matak\PycharmProjects\pythonProject\Animals_with_Attributes2\JPEGImages"  
OUTPUT_PATH = r"C:\Users\matak\PycharmProjects\pythonProject\Animals_with_Attributes2\JPEGImages\output" 
# Sınıf isimleri
CLASSES = ['collie', 'dolphin', 'elephant', 'fox', 'moose',
           'rabbit', 'sheep', 'squirrel', 'giant+panda', 'polar+bear']


def verify_data_directory():
    """Veri dizininin doğruluğunu kontrol et."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Veri dizini bulunamadı: {DATA_PATH}")

    missing_classes = []
    for class_name in CLASSES:
        class_path = os.path.join(DATA_PATH, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)

    if missing_classes:
        raise FileNotFoundError(
            f"Aşağıdaki sınıf dizinleri eksik: {', '.join(missing_classes)}")


def load_and_preprocess_data():
    """Veriyi yükle ve ön işleme uygula."""
    print(f"Veriler yükleniyor: {DATA_PATH}")
    verify_data_directory()

    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATA_PATH, class_name)
        image_files = sorted(os.listdir(class_path))[:MAX_IMAGES_PER_CLASS]
        print(f"Yüklenen görüntü sayısı - {class_name}: {len(image_files)}")

        for image_file in image_files:
            img_path = os.path.join(class_path, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Uyarı: Görüntü yüklenemedi {img_path}")
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(class_idx)

    return np.array(images), np.array(labels)


def create_cnn_model():
    """CNN modelini oluştur."""
    model = Sequential([
        # İlk Konvolüsyon Bloğu
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2)),

        # İkinci Konvolüsyon Bloğu
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Üçüncü Konvolüsyon Bloğu
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Yoğun Katmanlar
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_manipulated_images(images):
    """Farklı ışık koşulları uygula."""
    manipulated = []

    for img in images:
        # Parlaklığı artır
        bright = cv2.convertScaleAbs(img, alpha=1.5, beta=30)

        # Parlaklığı azalt
        dark = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)

        manipulated.append(bright / 255.0)  # Normalize

    return np.array(manipulated)


def get_wb_images(images):
    """Gray World renk sabitliği algoritmasını uygula."""
    wb_images = []

    for img in images:
        img = img.copy()

        # Her kanal için ortalama değerleri hesapla
        avg_b = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_r = np.mean(img[:, :, 2])

        # Ölçekleme faktörlerini hesapla
        avg_gray = (avg_b + avg_g + avg_r) / 3

        # Ölçekleme faktörlerini uygula
        img[:, :, 0] = np.clip(img[:, :, 0] * (avg_gray / avg_b), 0, 1)
        img[:, :, 1] = np.clip(img[:, :, 1] * (avg_gray / avg_g), 0, 1)
        img[:, :, 2] = np.clip(img[:, :, 2] * (avg_gray / avg_r), 0, 1)

        wb_images.append(img)

    return np.array(wb_images)


def main():
    # Çıktı dizinini oluştur
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Veriyi yükle ve ön işle
    print("Veriler yükleniyor ve ön işleniyor...")
    X, y = load_and_preprocess_data()

    # Etiketleri one-hot encoding formatına dönüştür
    y = to_categorical(y, NUM_CLASSES)

    # Veriyi eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=42
    )

    # Veri artırma generator'ını oluştur
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Modeli oluştur ve eğit
    print("Model oluşturuluyor ve eğitiliyor...")
    model = create_cnn_model()

    # Modeli veri artırma ile eğit
    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              steps_per_epoch=len(X_train) // BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, y_test))

    # Modeli kaydet
    model_path = os.path.join(OUTPUT_PATH, 'animal_classification_model.h5')
    model.save(model_path)
    print(f"Model kaydedildi: {model_path}")

    # Orijinal test seti üzerinde test et
    print("\nOrijinal test seti değerlendiriliyor...")
    original_score = model.evaluate(X_test, y_test)
    print(f"Orijinal Test Başarısı: {original_score[1] * 100:.2f}%")

    # Manipüle edilmiş görüntüler üzerinde test et
    print("\nManipüle edilmiş test seti değerlendiriliyor...")
    X_test_manipulated = get_manipulated_images(X_test)
    manipulated_score = model.evaluate(X_test_manipulated, y_test)
    print(f"Manipüle Edilmiş Test Başarısı: {manipulated_score[1] * 100:.2f}%")

    # Renk sabitliği uygulanmış görüntüler üzerinde test et
    print("\nRenk sabitliği uygulanmış test seti değerlendiriliyor...")
    X_test_wb = get_wb_images(X_test_manipulated)
    wb_score = model.evaluate(X_test_wb, y_test)
    print(f"Renk Sabitliği Uygulanmış Test Başarısı: {wb_score[1] * 100:.2f}%")

    # Sonuçları dosyaya kaydet
    results_path = os.path.join(OUTPUT_PATH, 'sonuclar.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Orijinal Test Başarısı: {original_score[1] * 100:.2f}%\n")
        f.write(f"Manipüle Edilmiş Test Başarısı: {manipulated_score[1] * 100:.2f}%\n")
        f.write(f"Renk Sabitliği Uygulanmış Test Başarısı: {wb_score[1] * 100:.2f}%\n")
    print(f"Sonuçlar kaydedildi: {results_path}")


if __name__ == "__main__":
    main()