import numpy as np
import cv2
import cv2.aruco as aruco


def load_camera_calibration(filepath):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("K").mat()
    distortion_coeffs = fs.getNode("D").mat()
    fs.release()
    return camera_matrix, distortion_coeffs


def draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    axis = np.float32([[length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)
    img_points, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    origin = tuple(np.int32(img_points[0].ravel()))
    x_axis = tuple(np.int32(img_points[1].ravel()))
    y_axis = tuple(np.int32(img_points[2].ravel()))

    frame = cv2.line(frame, origin, x_axis, (0, 0, 255), 5)  # Red X axis
    frame = cv2.line(frame, origin, y_axis, (0, 255, 0), 5)  # Green Y axis

    return frame


def track(matrix_coefficients, distortion_coefficients):
    cap = cv2.VideoCapture(0)  # 1: İkinci kamerayı kullan
    if not cap.isOpened():
        print("Kamera açılamadı. Doğru kamera indeksini kullandığından emin ol.")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    detector = aruco.ArucoDetector(aruco_dict)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamıyor.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = detector.detectMarkers(gray)

        if ids is not None:
            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], 0.02, matrix_coefficients, distortion_coefficients
                )
                aruco.drawDetectedMarkers(frame, corners)
                frame = draw_axis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

                # Markör ID'sini, rotasyon ve çeviri bilgisini ekranda yazdır
                cv2.putText(frame, f"ID: {ids[i][0]}", (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Rotasyon (rvec) ve çeviri (tvec) bilgilerini yazdır
                cv2.putText(frame, f"rvec: {rvec[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"tvec: {tvec[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('ArUco Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_matrix, distortion_coeffs = load_camera_calibration("C:\\Users\\MS\\Desktop\\camera_calibration.yml")
    track(camera_matrix, distortion_coeffs)
