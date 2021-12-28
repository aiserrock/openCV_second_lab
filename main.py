import imutils
import cv2

# минимальная площадь объекта для распознавания
min_area = 250

video_stream = cv2.VideoCapture('people1.mp4')

# очередь обрабатываемых кадров кадров
frames = []
queue_frames_size = 4

while True:
    # чтение информации о кадре
    frame = video_stream.read()[1]

    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    # Приводим кадр к оттенкам серого(черынй и белый)
    # т.к в этом цветовом пространстве обработка кадра происходит проще
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # добавил блюр для еще большего упрощения расчетов
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # проверяю что очередь сформирована нужного мне размера
    if len(frames) < queue_frames_size:
        frames.append(gray)
        continue

    # ищем разницу между первым кадром и кадром перед gray (5 кадр)
    frameDelta = cv2.absdiff(frames[0], gray)
    # бинаризация
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # блюр
    thresh = cv2.dilate(thresh, None, iterations=3)
    # ищем контуры обьектов
    conturs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conturs = imutils.grab_contours(conturs)

    for c in conturs:
        if cv2.contourArea(c) < min_area:
            continue
        # запомнить окрамляющий прямоугольник для обьекта
        (x, y, w, h) = cv2.boundingRect(c)
        if w/h < 0.65:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # производим смещение кадра для след. сравнения путем удаления первого
    del frames[0]
    frames.append(gray)

    # 4 вида камер
    cv2.imshow("Origin", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Gray", gray)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_stream.release()
cv2.destroyAllWindows()
