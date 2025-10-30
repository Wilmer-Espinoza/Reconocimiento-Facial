import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("No se puede acceder a la cámara.")
else:
    print("Cámara abierta con éxito.")

cap.release()