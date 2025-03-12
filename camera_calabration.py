import numpy as np
import cv2
import glob
import argparse

# Kamera kalibrasyonu için kullanılan kriterler
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, prefix, image_format, square_size, width, height):
    """ Satranç tahtası resimlerini kullanarak kamera kalibrasyonu yapar. """
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Kare boyutu ile çarpıyoruz.

    objpoints = []  # 3D gerçek dünya noktaları
    imgpoints = []  # 2D görüntü noktaları

    images = glob.glob(dirpath + '/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Satranç tahtası köşelerini bul
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Görüntüye köşeleri çiz
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    # Kamerayı kalibre et
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


def save_coefficients(mtx, dist, path):
    """ Kamera matrisini ve bozulma katsayılarını kaydeder. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()


def load_coefficients(path):
    """ Daha önce kaydedilmiş kamera kalibrasyon değerlerini yükler. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kamera Kalibrasyonu')
    parser.add_argument('--image_dir', type=str, required=True, help='Resimlerin olduğu klasör')
    parser.add_argument('--image_format', type=str, required=True, help='Resim formatı (png/jpg)')
    parser.add_argument('--prefix', type=str, required=True, help='Resimlerin ön eki (örneğin, "calib_")')
    parser.add_argument('--square_size', type=float, required=True, help='Satranç tahtası karesi boyutu (m)')
    parser.add_argument('--width', type=int, required=True, help='Satranç tahtasının genişliği (iç köşe sayısı)')
    parser.add_argument('--height', type=int, required=True, help='Satranç tahtasının yüksekliği (iç köşe sayısı)')
    parser.add_argument('--save_file', type=str, required=True,
                        help='Kalibrasyon sonuçlarının kaydedileceği YML dosyası')

    args = parser.parse_args()

    ret, mtx, dist, rvecs, tvecs = calibrate(args.image_dir, args.prefix, args.image_format, args.square_size,
                                             args.width, args.height)
    save_coefficients(mtx, dist, args.save_file)
    print("Kalibrasyon tamamlandı. RMS hatası:", ret)
