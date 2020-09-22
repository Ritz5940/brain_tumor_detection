from tkinter import *
import tkinter
import tkinter as tk
from tkinter import filedialog 
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np;
from skimage import data
from skimage.feature import greycomatrix, greycoprops
import skimage.io
import skimage.feature
import copy
# Create a window
window =Tk()
window.title("BRAIN TUMOR DETECTION USING MACHINE LEARNING")
window.geometry("1500x1700")
message = tk.Label(window, text="BRAIN TUMOR DETECTION USING MACHINE LEARNING" ,bg="Green"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 

message.place(x=200, y=20)
def answer():
    global cv_img, mask,img,canvas1,canvas4
    canvas1.delete("all")
    canvas4.delete("all")
    #showerror("Answer", "Sorry, no answer available")
    name= filedialog.askopenfilename() 
    cv_img = cv2.imread(name)
    img=cv_img
    
    mask = np.zeros(cv_img.shape[:2], dtype="uint8") * 255
    # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
    height, width, no_channels = cv_img.shape

    # Create a canvas that can fit the above image
    canvas1 = tkinter.Canvas(window, width = width, height = height)
    canvas1.place(x=50,y=200)
     
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
     
    # Add a PhotoImage to the Canvas
    canvas1.create_image(0, 0, image=photo, anchor=tkinter.NW)
     
    # Run the window loop
    window.mainloop()
def rgb2gray():
    global cv_img,img_gray,canvas1,canvas2
    img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    height, width= img_gray.shape
    canvas1.delete("all")
    # Create a canvas that can fit the above image
    canvas2 = tkinter.Canvas(window, width = width, height = height)
    canvas2.place(x=50,y=200)
     
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img_gray))
     
    # Add a PhotoImage to the Canvas
    canvas2.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
def Thresholding():
    global cv_img,img_gray,im_th,canvas2,canvas3
    canvas2.delete("all")
    for threshold in range(150,180):
        ret,im_th= cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
        height, width= im_th.shape

        # Create a canvas that can fit the above image
        canvas3 = tkinter.Canvas(window, width = width, height = height)
        canvas3.place(x=50,y=200)
         
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(im_th))
         
        # Add a PhotoImage to the Canvas
        canvas3.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
def detect():
    global im_th,mask,cv_img,img,clone_img,canvas3,canvas4
    canvas3.delete("all")
    clone_img = copy.copy(img)
    cv2.imshow('median filter1',img)
    denoised=cv2.medianBlur(im_th,9)
    cv2.imshow('median filter',denoised)
    edges = cv2.Canny(denoised,100,200)
    cv2.imshow('median filter3',edges)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(denoised,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(cv_img,markers)
    cv_img[markers == -1] = [255,0,0]
    #markers[unknown==255] = 0
    markers1 = cv2.watershed(clone_img,markers)
    clone_img[markers1 == 1] = [0,0,0]
    height, width= mask.shape

    # Create a canvas that can fit the above image
    canvas4 = tkinter.Canvas(window, width = width, height = height)
    canvas4.place(x=50,y=200)
     
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
     
    # Add a PhotoImage to the Canvas
    canvas4.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
def Feature_extraction():
    global clone_img
    im_th=cv2.cvtColor(clone_img, cv2.COLOR_BGR2GRAY)
    g = skimage.feature.greycomatrix(im_th, [1], [0], levels=256, symmetric=False, 
        normed=True)
    contrast = skimage.feature.greycoprops(g, 'contrast')[0][0]
    energy = skimage.feature.greycoprops(g, 'energy')[0][0]
    homogeneity = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
    correlation = skimage.feature.greycoprops(g, 'correlation')[0][0]
    lbl = tk.Label(window, text="CONTRAST",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
    lbl.place(x=400, y=600)
    lbl = tk.Label(window, text="ENERGY",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
    lbl.place(x=400, y=700)
    lbl = tk.Label(window, text="HOMOGENEITY",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
    lbl.place(x=400, y=800)
    lbl = tk.Label(window, text="CORRELATION",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
    lbl.place(x=400, y=900)
    message = tk.Label(window, text=contrast ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
    message.place(x=700, y=600)
    message = tk.Label(window, text=energy ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
    message.place(x=700, y=700)
    message = tk.Label(window, text=homogeneity ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
    message.place(x=700, y=800)
    message = tk.Label(window, text=correlation ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
    message.place(x=700, y=900)
    print(contrast)
    print(energy)
    print(homogeneity)
    print(correlation)
    window.mainloop()
canvas1 = tkinter.Canvas(window, width = 300, height = 400)
canvas1.place(x=50,y=200)
canvas4 = tkinter.Canvas(window, width = 300, height = 400)
canvas4.place(x=50,y=200)

btn1=Button(text='Load Image', command=answer,width=10  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn1.place(x=1200,y=200)
btn2=Button(text='Rgb2gary', command=rgb2gray,width=10  ,height=2 ,activebackground = "Green" ,font=('times', 15, ' bold '))
btn2.place(x=1200,y=300)
btn3=Button(text='Gray Thr', command=Thresholding,width=10  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn3.place(x=1200,y=400)
btn4=Button(text='find Tumor', command=detect,width=10  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn4.place(x=1200,y=500)
btn5=Button(text='Feature Extraction',command=Feature_extraction,width=15  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn5.place(x=1200,y=600)
# Load an image using OpenCV

