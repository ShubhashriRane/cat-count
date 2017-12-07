#!/usr/bin/env python

from __future__ import print_function
from firebase import firebase

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,1:] += rects[:,:1]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalcatface.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
   
    cam = create_capture(video_src)
    count=0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t=clock()
        rects = detect(gray, cascade)
        vis = img.copy()
       
        rects = cascade.detectMultiScale(gray, scaleFactor=1.3,
	minNeighbors=10, minSize=(75, 75))


        for (i, (x, y, w, h)) in enumerate(rects):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, "Cat #{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
	
        cv2.imshow('cat', vis)

        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
    firebase = firebase.FirebaseApplication('https://mitdb-7e16b.firebaseio.com/')
    data = {'Cat' : count }

    result= firebase.put('https://mitdb-7e16b.firebaseio.com','/mitdb-7e16b',data)
    print(result)    
