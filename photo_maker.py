import cv2
import time

# Kamerayı aç
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

foto_num = 1  # Fotoğraf numarasını takip etmek için

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare okunamadı!")
        break

    cv2.imshow("Webcam", frame)

    # 's' tuşuna basınca fotoğraf çek ve farklı isimle kaydet
    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = f"foto_{foto_num}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Fotoğraf kaydedildi: {filename}")
        foto_num += 1  # Fotoğraf numarasını artır

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
