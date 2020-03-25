import cv2
import numpy as np
import cnn
from player import Player

class Pitch:

    line_history = []
    tags = []
    players = []

    def __init__(self, tags):
        self.tags = tags

    def update(self, frame, detect=False):
        h, w, _ = frame.shape
        lines = self.fetchLines(frame)
        tagged_lines = []

        if self.line_history == []:
            for i in range(len(lines)):
                tagged_lines.append([lines[i], self.tags[i]])

        else:
            for line, tag in self.line_history:
                if tag == 'I':
                    continue

                # calculate the y value at the midle of the screen
                middle_int = line['grad'] * w/2 + line['yint']

                # See if there is a newline close to each
                similar_lines = [line2 for line2 in lines if abs(line2['grad'] - line['grad']) < 0.1 and abs(middle_int - (line2['grad'] * w/2 + line2['yint'])) < 20]
                if len(similar_lines) >= 1:
                    lines.remove(similar_lines[0])
                    tagged_lines.append([similar_lines[0], tag])
                else:
                    print('Missed')

        if self.line_history != []:
            for line in lines:
                tagged_lines.append([line, 'I'])
        self.line_history = tagged_lines

        if detect:
            self.players = self.findPlayers(frame)
        else:
            for player in self.players:
                player.track(frame)

    def annotate(self, frame):
        self.drawLines(self.line_history, frame)

    def findPlayers(self, frame):
        results = cnn.detect(frame)
        results = zip(results[0]['rois'], results[0]['class_ids'])
        players = []
        for box, class_id in results:
            if class_id != 1:
                continue
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0,0,255), 2)
            players.append(Player([box[1], box[0], box[3], box[2]], frame))

        return players

    def fetchLines(self, frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask = self.findPitchMask(frame)
        gray = cv2.bitwise_and(gray,gray, mask= mask)

        edges = cv2.Canny(gray,20,150,apertureSize = 3)

        kernel = np.ones((7,7), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        minLineLength = 350
        maxLineGap = 20
        lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180,threshold = 500, minLineLength = minLineLength, maxLineGap = maxLineGap)
        lines = [l[0] for l in lines]

        params = []
        edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        line_eqs = [self.findEquationOfLine((x1,y1), (x2,y2)) for x1,y1,x2,y2 in lines]

        filtered_line_eqs = []

        while len(line_eqs) != 0:
            line = line_eqs.pop()
            similar_lines = [line2 for line2 in line_eqs if abs(line['grad'] - line2['grad']) < 0.05 and abs(line['yint'] - line2['yint']) < 6]

            [line_eqs.remove(v) for v in similar_lines]

            m = (line['grad'] + sum([line2['grad'] for line2 in similar_lines])) / float(len(similar_lines) + 1)
            c = (line['yint'] + sum([line2['yint'] for line2 in similar_lines])) / float(len(similar_lines) + 1)

            filtered_line_eqs.append({'grad': m, 'yint': c})

        # Sort by y intercept
        filtered_line_eqs = sorted(filtered_line_eqs, key=lambda x: x['yint'])

        return filtered_line_eqs

    def findPitchMask(self, frame):
        for i in range(500):
            frame = cv2.blur(frame,(9,9))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of white color in HSV
        # change it according to your need !
        lower_white = np.array([25, 52, 72])
        upper_white = np.array([102, 255, 255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)

        return mask

    def findEquationOfLine(self, p1, p2):
        x1,y1 = p1
        x2,y2 = p2
        m = (float(y2)-y1)/(float(x2)-x1)

        c = y1 - m * x1

        return {'grad': m, 'yint': c}

    def drawLines(self, tagged_lines, frame):
        h, w, _ = frame.shape
        for line, tag in tagged_lines:
            p1 = (0, int(line['yint']))
            p2 = (w, int(line['grad']*w + line['yint']))
            cv2.line(frame, p1, p2, self.color(tag), 4)

    def color(self, tag):
        if tag in ['BL', 'SL']:
            return (255, 0, 0)
        elif tag in ['20M', 'TL']:
            return (0, 255, 0)
