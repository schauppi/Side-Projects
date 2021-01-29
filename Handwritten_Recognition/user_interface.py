from tkinter import*
import PIL
from PIL import Image, ImageDraw, ImageOps, ImageTk
from numpy import asarray
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt



window = Tk()
window.title("Handwritten Digit Recognition")
window.geometry("500x300")

def paint(event):
    #Mouse coordinates here
    x1 = event.x
    y1 = event.y

    #Draw the image
    paint_job.create_rectangle((x1, y1)*2, fill="blue")
    coordinate_list.append((x1, y1))
    image1 = draw.line((x1, y1)*2, fill="white")

def save_image():
    #Convert drawn Image to 28x28 Pixels
    resized_image = image1.resize((28,28))
    converted_image = resized_image.convert('L')
    filename = "image.png"
    converted_image.save(filename)
    predict()

def clear_image():
    paint_job.delete("all")

def predict():
    model_hw = keras.models.load_model("handwritten_recognition_model.h5")
    im = Image.open("image.png")
    im = np.resize(im, (28, 28, 1))
    im2arr = np.array(im)
    im2arr = im2arr.reshape(1, 28, 28)
    x_pred = im2arr / 255.0
    pred = model_hw.predict(x_pred)
    histo(pred)

def histo(pred):
    predlist = []
    for i in range(len(pred[0])):
        predlist.append(pred[0][i])
    x = np.arange(len(predlist))
    plt.bar(x, height=[predlist[0], predlist[1], predlist[2], predlist[3], predlist[4], predlist[5],
                       predlist[6], predlist[7], predlist[8], predlist[9]])
    plt.xticks(x, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.savefig("histo.png")
    open_histo_image()

def open_histo_image():
    img = Image.open("histo.png")
    img = img.resize((200, 200),Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.image = img
    panel.place(x=250, y=25)


coordinate_list = []

paint_job = Canvas(window, width=100, height=100, bg="red")
paint_job.place(x=150, y=100, anchor="center")

image1 = PIL.Image.new('RGB', (100, 100), "black")
draw = ImageDraw.Draw(image1)

paint_job.bind("<B1-Motion>", paint)

submit_button = Button(window, text="Submit", width=10, command=save_image)
submit_button.place(x=150, y=170, anchor="center")

clear_button = Button(window, text="Clear", width=10, command=clear_image)
clear_button.place(x=150, y=200, anchor="center")

window.mainloop()

