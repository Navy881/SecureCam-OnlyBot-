# -*- coding: utf-8 -*-

import os
import cv2
import imutils
import numpy as np
from datetime import datetime

from src.tools.video_record import record, save_image


class Camera(object):
    def __init__(self, cam, fps):
        self.video = cv2.VideoCapture(cam)
        self.video.set(cv2.CAP_PROP_FPS, fps)
        # self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def make_screenshot(self):
        # Чтение видеоканала cap, frame - изображение
        ret, frame = self.video.read()
        if not ret:
            print("Sorry, webcam is not found")
        else:
            cv2.imwrite('photo/bot_screenshot.png', frame)

    def motion_detect(self, running, video_file, show_edges,
                      dnn_detection_status, net, classes, colors, given_confidence=0.2,
                      min_area=10, blur_size=11, blur_power=1, threshold_low=50, sending_period=60):
        # Нижний порог площади контура
        first_frame = None
        
        fps = self.video.get(cv2.CAP_PROP_FPS)

        # Пока есть сигнал из видеоканала
        while running:
            # Чтение видеоканала cap, frame - изображение
            ret, frame = self.video.read()
            text = "Unoccupied"
            if not ret:
                print("Sorry, webcam is not found")
                break

            # Изменение размера изображения
            frame1 = imutils.resize(frame, width=500)
            # Обесцвечивание изображения
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            # Размытие обесцвеченного изображения
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), blur_power)

            # Первым кадром видео должен быть размытый кадр    
            # firstFrame = gray
            if first_frame is None:
                first_frame = gray
                continue
            # Разница между первым кадром и обесцвеченным следующим кадром из видеоряда
            frame_delta = cv2.absdiff(first_frame, gray)
            # Бинаризация разницы
            thresh = cv2.threshold(frame_delta, threshold_low, 255, cv2.THRESH_BINARY)[1]
            # Подамвления шума
            thresh = cv2.dilate(thresh, None, iterations=2)
            # Выделение замкнутых контуров
            _, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.putText(frame, "Camera 1", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, datetime.now().strftime("%d-%m-%Y %H:%M:%S%p"), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Цикл по найденным контурам
            for c in cnts:
                # Если площадь котура меньше чем нижний порог
                if cv2.contourArea(c) < min_area:
                    continue
                # Координаты, длина и ширина контура
                (x, y, w, h) = cv2.boundingRect(c)
                # Рисование рамки контура на изображении
                if show_edges:
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = "Occupied"
                # Обновление первого кадра
                # first_frame = gray

                # Если найденный контур больше или равен значению нижнего порога
                if cv2.contourArea(c) >= min_area:
                    # Зависать изображение в видеофайл
                    record(video_file, frame)

                    # Детектирование объектов
                    if dnn_detection_status:
                        frame1 = self.real_time_detection(frame1, net, classes, colors, given_confidence)

                    # Если кнопка бота включена и время после последней отправки больше заданного периода
                    if os.path.exists('photo/screenshot_temp.png'):
                        file_create_time = os.path.getmtime('photo/screenshot_temp.png')
                    else:
                        file_create_time = 0
                    send_delta = datetime.today().timestamp() - file_create_time
                    if int(send_delta) > sending_period:
                        save_image(frame1)  # cv2.imwrite("photo/screenshot_temp.png", frame1)

                else:
                    video_file.release()
                
            cv2.putText(frame1, "Camera 1 {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame1, datetime.now().strftime("%d-%m-%Y %H:%M:%S%p"), (10, frame1.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame1, "FPS: "+str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Перевод видео в изображение для отображения в вебе
            # ret, jpeg = cv2.imencode('.jpg', frame1)

            # return frame1, jpeg.tobytes(), text
            return frame1, text

    @staticmethod
    def real_time_detection(frame, net, classes, colors, given_confidence):
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > given_confidence:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        return frame

    
    def real_time_detection_2(self, dnn_detection_status, net, classes, colors, given_confidence):
        while dnn_detection_status:
            # Чтение видеоканала cap, frame - изображение
            ret, frame = self.video.read()
            # frame = imutils.resize(frame, width=400)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > given_confidence:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

            return frame
    
