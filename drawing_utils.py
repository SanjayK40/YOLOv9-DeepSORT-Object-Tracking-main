import cv2
import numpy as np
from collections import deque

def classNames():
    return ["psychiatrist", "child"]

def colorLabels(classid):
    if classid == 0: # therapist
        color = (85, 222, 255)
    elif classid == 1: # child
        color = (222, 82, 175)
    else:
        color = (200, 100, 0)
    return tuple(color)

def draw_boxes(frame, bbox_xyxy, identities=None, categories=None, offset=(0, 0)):
    height, width, _ = frame.shape
    className = classNames()
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        y1 += offset[1]
        x2 += offset[0]
        y2 += offset[1]

        center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cat = int(categories[i]) if categories is not None else 0
        color = colorLabels(cat)
        id = int(identities[i]) if identities is not None else 0

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = className[cat]
        label = f"{name}:{id}"
        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, 2, (0, 255, 0), cv2.FILLED)

    return frame
