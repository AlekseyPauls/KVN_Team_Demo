import logging
import logging.handlers
import queue
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import os
import face_recognition
import time

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)


def main():
    st.header("KVN Team demo")

    realtime_face_recognition_page = (
        "Realtime Face Recognition"
    )
    realtime_ppe_detection_page = (
        "Realtime Personal Protective Equipment detection"
    )

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            realtime_face_recognition_page,
            realtime_ppe_detection_page
        ],
    )
    st.subheader(app_mode)

    if app_mode == realtime_face_recognition_page:
        app_realtime_face_recognition()
    elif app_mode == realtime_ppe_detection_page:
        app_realtime_ppe_detection()
    elif app_mode == streaming_face_recognition_page:
        app_streaming_face_recognition()
    elif app_mode == streaming_ppe_detection_page:
        app_streaming_ppe_detection()


def app_realtime_face_recognition():

    class FaceRecognitionTransformer(VideoTransformerBase):

        def __init__(self) -> None:
            self.known_face_encodings = []
            self.known_face_names = []
            self.cur_frame = 0
            self.face_names = []

            for face_name in os.listdir(os.path.join("data", "faces")):
                face = face_recognition.load_image_file(os.path.join("data", "faces", face_name))
                face_encoding = face_recognition.face_encodings(face)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(face_name[:face_name.rfind(".")])

            self.result_queue = queue.Queue()

        def _annotate_image(self, image, face_locations, face_names):
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                if name != "Unknown":
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                else:
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            return image, self.face_names

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            start = time.time()

            image = frame.to_ndarray(format="bgr24")
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

            face_locations = face_recognition.face_locations(small_frame)

            self.cur_frame += 1
            print(self.cur_frame)

            if self.cur_frame % 10 == 0:
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                self.face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    self.face_names.append(name)

            annotated_image, result = self._annotate_image(image, face_locations, self.face_names)
            self.result_queue.put(result)

            end = time.time()
            print("Transform time: ", str(end - start))

            return annotated_image

    webrtc_streamer(
        key="face-recognition",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=FaceRecognitionTransformer,
        async_transform=True,
    )


def app_realtime_ppe_detection():
    class PpeDetectionTransfromer(VideoTransformerBase):
        def __init__(self) -> None:
            self.frame_cur = 0
            self.bounding_boxes = []
            self.confidences = []
            self.class_numbers = []

            self.network = cv2.dnn.readNetFromDarknet('./yolo/yolov3_ppe_test.cfg',
                                                      './yolo/yolov3_ppe_train_9000.weights')

            self.layers_names_all = self.network.getLayerNames()
            self.layers_names_output = [self.layers_names_all[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]

            with open('yolo/class.names') as f:
                self.labels = [line.strip() for line in f]

            self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
            self.results = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            probability_minimum = 0.7
            threshold = 0.3
            bounding_boxes = []
            confidences = []
            class_numbers = []

            frame = frame.to_ndarray(format="bgr24")
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            h, w = small_frame.shape[:2]

            if self.frame_cur == 0 or self.frame_cur % 5 == 0:
                blob = cv2.dnn.blobFromImage(small_frame, 1 / 255.0, (416, 416),
                                             swapRB=True, crop=False)
                self.network.setInput(blob)
                output_from_network = self.network.forward(self.layers_names_output)

                for result in output_from_network:

                    for detected_objects in result:
                        scores = detected_objects[5:]
                        class_current = np.argmax(scores)
                        confidence_current = scores[class_current]
                        if confidence_current > probability_minimum:
                            box_current = detected_objects[0:4] * np.array([w, h, w, h])
                            x_center, y_center, box_width, box_height = box_current
                            x_min = int(x_center - (box_width / 2))
                            y_min = int(y_center - (box_height / 2))
                            bounding_boxes.append([x_min, y_min,
                                                   int(box_width), int(box_height)])
                            confidences.append(float(confidence_current))
                            class_numbers.append(class_current)
                        self.bounding_boxes = bounding_boxes
                        self.confidences = confidences
                        self.class_numbers = class_numbers

                self.results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

                if len(self.results) > 0:
                    for i in self.results.flatten():
                        x_min, y_min = 4 * bounding_boxes[i][0], 4 * bounding_boxes[i][1]
                        box_width, box_height = 4 * bounding_boxes[i][2], 4 * bounding_boxes[i][3]
                        colour_box_current = self.colours[class_numbers[i]].tolist()

                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_min + box_width, y_min + box_height),
                                      colour_box_current, 2)

                        text_box_current = '{}: {:.4f}'.format(self.labels[int(class_numbers[i])],
                                                               confidences[i])

                        cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

            else:
                if len(self.results) > 0:
                    for i in self.results.flatten():
                        x_min, y_min = 4 * self.bounding_boxes[i][0], 4 * self.bounding_boxes[i][1]
                        box_width, box_height = 4 * self.bounding_boxes[i][2], 4 * self.bounding_boxes[i][3]
                        colour_box_current = self.colours[self.class_numbers[i]].tolist()

                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_min + box_width, y_min + box_height),
                                      colour_box_current, 2)

                        text_box_current = '{}: {:.4f}'.format(self.labels[int(self.class_numbers[i])],
                                                               self.confidences[i])

                        cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

            self.frame_cur += 1

            return frame

    webrtc_streamer(
        key="real-ppe-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=PpeDetectionTransfromer,
        async_transform=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(filename)s:%(lineno)d: "
               "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    main()
