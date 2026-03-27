import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Kamera açılamadı! İzinleri veya indeks numarasını kontrol edin.")
    exit()

print("Kamera başarıyla açıldı. Kapatmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    cv2.imshow("Kamera Testi", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()