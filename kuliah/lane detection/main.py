import cv2
import numpy as np

def gambar_garis(gambar, garis):
    if garis is None:
        return gambar
    
    gambar_garis = np.zeros_like(gambar)
    for garis_tunggal in garis:
        for x1, y1, x2, y2 in garis_tunggal:
            cv2.line(gambar_garis, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
    
    return cv2.addWeighted(gambar, 0.8, gambar_garis, 1, 0)

def wilayah_minat(gambar, titik_wilayah):
    masker = np.zeros_like(gambar)
    cv2.fillPoly(masker, [titik_wilayah], 255)
    return cv2.bitwise_and(gambar, masker)

def deteksi_jalur(gambar):
    tinggi, lebar = gambar.shape[:2]
    
    gambar_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    gambar_blur = cv2.GaussianBlur(gambar_abu, (5, 5), 0)
    gambar_canny = cv2.Canny(gambar_blur, 50, 150)
    
    titik_wilayah_minat = np.array([
        [0, tinggi],
        [lebar // 2, int(tinggi * 0.6)],
        [lebar, tinggi]
    ], dtype=np.int32)
    
    gambar_terpotong = wilayah_minat(gambar_canny, titik_wilayah_minat)
    
    garis = cv2.HoughLinesP(
        gambar_terpotong,
        rho=2,
        theta=np.pi/180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )
    
    return gambar_garis(gambar, garis)

def proses_video(nama_file):
    video = cv2.VideoCapture(nama_file)
    
    if not video.isOpened():
        print("Error: Tidak dapat membuka video.")
        return
    
    while True:
        berhasil_ambil, frame = video.read()
        
        if not berhasil_ambil:
            break
        
        frame_terproses = deteksi_jalur(frame)
        
        cv2.imshow('Deteksi Jalur', frame_terproses)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    proses_video('lane_detection_video.mp4')