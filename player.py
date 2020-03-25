import cv2

class Player:
    tracker = None

    def __init__(self, box, frame):
        bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)

    def track(self, frame):
        _, box = self.tracker.update(frame)
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2)

        return box
