from tkinter import filedialog
import tkinter as tk
from tkinter import *
import cv2
import numpy as np
import glob
import random

dir = None
def open_folder():
    dir = filedialog.askdirectory()
    print(dir)
    x=dir.replace("/","\\")
    x=x+"\*.jpg"
    print(x)
    f(x)

root = tk.Tk()
root.geometry('800x600')
root.configure(bg = 'lightsteelblue')
root.title("Object Detection")


def f(pat):
    # Load Yolo
    net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")

    # Name custom object
    classes = ["Guns","Knife"]

    # Images path
    images_path = glob.glob(pat)



    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Insert here the path of your images
    random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
                    print(confidence)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]) + str(" : ")
                na = str(int(confidences[i]*100)) + str(" %")
                #color = colors[class_ids[i]]
                color = (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label + na, (x, y - 10), font, 0.5, color, 2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()

#Title
tk.Label(
        text='Object Detection',
        pady = '20',
        foreground = 'steelblue',
        background = 'lightsteelblue',
        font = "Verdana 35 bold"
        ).pack(side = 'top')

#picture
canvas = Canvas(root, width = 350, height = 350,)
canvas.pack()
img = PhotoImage(file="bg.png")
canvas.create_image(0,0, anchor=NW, image=img)

#Button
tk.Button(
        root,
        text='Open',
        bg = 'steelblue',
        width = '20',
        font = "Verdana 12 bold",
        command=open_folder
        ).pack()

#Sub-title
tk.Label(
        text="Please select directory\n\n",
        foreground = 'red',
        background = 'lightsteelblue',
        font = "times 10 bold"
        ).pack(side = 'bottom')

root.mainloop()
