import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import glob
import matplotlib.pyplot as plt
from scipy import stats

def preprocess_image(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    # Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalisasi
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return normalized, img

def segment_image(image):
    # Terapkan Otsu thresholding
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def calculate_statistics(images, labels):
    # Hitung probabilitas kelas
    total_samples = len(labels)
    class_probabilities = {
        'shingles': np.sum(labels == 1) / total_samples,
        'healthy': np.sum(labels == 0) / total_samples
    }
    
    # Hitung variabilitas dalam kelas
    shingles_features = images[labels == 1]
    healthy_features = images[labels == 0]
    
    within_class_variance = {
        'shingles': np.var(shingles_features, axis=0),
        'healthy': np.var(healthy_features, axis=0)
    }
    
    # Hitung variabilitas antar kelas
    mean_shingles = np.mean(shingles_features, axis=0)
    mean_healthy = np.mean(healthy_features, axis=0)
    between_class_variance = np.var([mean_shingles, mean_healthy], axis=0)
    
    # Hitung varians total
    total_variance = np.var(images, axis=0)
    
    return {
        'class_probabilities': class_probabilities,
        'within_class_variance': within_class_variance,
        'between_class_variance': between_class_variance,
        'total_variance': total_variance
    }

def plot_histogram_and_threshold(gray_img, binary_img, title):
    plt.figure(figsize=(15, 5))
    
    # Plot histogram
    plt.subplot(121)
    plt.hist(gray_img.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.title(f'Histogram - {title}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')
    
    # Plot binary image
    plt.subplot(122)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Otsu Thresholding Result')
    plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def plot_thresholding_results(original_img, gray_img, binary_img, title):
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image - {title}')
    plt.axis('off')
    
    # Plot grayscale image
    plt.subplot(132)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    # Plot binary image after Otsu thresholding
    plt.subplot(133)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Otsu Thresholding Result')
    plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def extract_features(binary_image):
    # Hitung area
    area = np.sum(binary_image == 255)
    # Hitung perimeter
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    # Hitung circularity
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    # Hitung mean intensity
    mean_intensity = np.mean(binary_image)
    
    return [area, perimeter, circularity, mean_intensity]

def load_and_process_data(base_path):
    X = []
    y = []
    
    # Proses gambar shingles
    shingles_path = os.path.join(base_path, 'train-shingles', '*.*')
    shingles_images = []
    for img_path in glob.glob(shingles_path):
        try:
            # Preprocessing
            processed_img, original_img = preprocess_image(img_path)
            # Segmentasi
            segmented = segment_image(processed_img)
            # Ekstraksi fitur
            features = extract_features(segmented)
            X.append(features)
            y.append(1)  # Label untuk shingles
            
            # Simpan gambar untuk visualisasi
            if len(shingles_images) < 3:  # Ambil 3 gambar pertama
                shingles_images.append((original_img, processed_img, segmented))
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Proses gambar healthy
    healthy_path = os.path.join(base_path, 'train-Healthy', '*.*')
    healthy_images = []
    for img_path in glob.glob(healthy_path):
        try:
            processed_img, original_img = preprocess_image(img_path)
            segmented = segment_image(processed_img)
            features = extract_features(segmented)
            X.append(features)
            y.append(0)  # Label untuk healthy
            
            # Simpan gambar untuk visualisasi
            if len(healthy_images) < 3:  # Ambil 3 gambar pertama
                healthy_images.append((original_img, processed_img, segmented))
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return np.array(X), np.array(y), shingles_images, healthy_images

def main():
    # Path ke dataset
    base_path = 'shingles_segmentation_dataset/train_set'
    
    # Load dan proses data
    print("Memuat dan memproses data...")
    X, y, shingles_images, healthy_images = load_and_process_data(base_path)
    
    # Hitung statistik
    print("\nMenghitung statistik...")
    stats_results = calculate_statistics(X, y)
    
    # Tampilkan hasil statistik
    print("\nHasil Statistik:")
    print("\nProbabilitas Kelas:")
    for class_name, prob in stats_results['class_probabilities'].items():
        print(f"{class_name}: {prob:.4f}")
    
    print("\nVariabilitas dalam Kelas:")
    for class_name, variance in stats_results['within_class_variance'].items():
        print(f"{class_name}: {variance}")
    
    print("\nVariabilitas antar Kelas:")
    print(stats_results['between_class_variance'])
    
    print("\nVarians Total:")
    print(stats_results['total_variance'])
    
    # Visualisasi hasil thresholding
    print("\nMenampilkan hasil thresholding...")
    
    # Plot untuk gambar shingles
    for i, (original, gray, binary) in enumerate(shingles_images):
        # Plot thresholding results
        fig = plot_thresholding_results(original, gray, binary, f'Shingles Sample {i+1}')
        plt.savefig(f'shingles_thresholding_{i+1}.png')
        plt.close()
        
        # Plot histogram
        fig = plot_histogram_and_threshold(gray, binary, f'Shingles Sample {i+1}')
        plt.savefig(f'shingles_histogram_{i+1}.png')
        plt.close()
    
    # Plot untuk gambar healthy
    for i, (original, gray, binary) in enumerate(healthy_images):
        # Plot thresholding results
        fig = plot_thresholding_results(original, gray, binary, f'Healthy Sample {i+1}')
        plt.savefig(f'healthy_thresholding_{i+1}.png')
        plt.close()
        
        # Plot histogram
        fig = plot_histogram_and_threshold(gray, binary, f'Healthy Sample {i+1}')
        plt.savefig(f'healthy_histogram_{i+1}.png')
        plt.close()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nMelatih model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nHasil Evaluasi:")
    print(f"Akurasi: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main() 