from tkinter import *
import tkinter
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar 
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np;
import copy
from skimage import data
from skimage.feature import greycomatrix, greycoprops
import skimage.io
import skimage.feature
import xlwt 
from xlwt import Workbook
import os
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time 
# Create a window
window =Tk()
window.title("BRAIN TUMOR DETECTION USING MACHINE LEARNING")
window.geometry("1366x768")
message = tk.Label(window, text="BRAIN TUMOR DETECTION USING MACHINE LEARNING" ,bg="Green"  ,fg="white"  ,width=50  ,height=3,font=('times', 20, 'italic bold underline')) 

message.place(x=270, y=20)
def answer():
    global cv_img, mask,img,canvas1,canvas4,canvas2,canvas3
    canvas1.delete("all")
    canvas2.delete("all")
    canvas3.delete("all")
    canvas4.delete("all")
    #showerror("Answer", "Sorry, no answer available")
    name= filedialog.askopenfilename() 
    cv_img = cv2.imread(name)
    newsize = (180, 218) 
    cv_img = cv2.resize(cv_img,newsize) 
    img=cv_img
    
    mask = np.zeros(cv_img.shape[:2], dtype="uint8") * 255
    # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
    height, width, no_channels = cv_img.shape

    lbl = tk.Label(window, text="Original Image",width=20  ,height=2  ,fg="black",font=('times', 10, ' bold ') ) 
    lbl.place(x=70, y=160)
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
    #canvas1.delete("all")
    lbl = tk.Label(window, text="Gray Image",width=20  ,height=2  ,fg="black",font=('times', 10, ' bold ') ) 
    lbl.place(x=270, y=160)
    # Create a canvas that can fit the above image
    canvas2 = tkinter.Canvas(window, width = width, height = height)
    canvas2.place(x=250,y=200)
     
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img_gray))
     
    # Add a PhotoImage to the Canvas
    canvas2.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
def Thresholding():
    global cv_img,img_gray,im_th,canvas2,canvas3
    #canvas2.delete("all")
    for threshold in range(150,180):
        ret,im_th= cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
        height, width= im_th.shape
        lbl = tk.Label(window, text="Threshold Image",width=20  ,height=2  ,fg="black",font=('times', 10, ' bold ') ) 
        lbl.place(x=470, y=160)
        # Create a canvas that can fit the above image
        canvas3 = tkinter.Canvas(window, width = width, height = height)
        canvas3.place(x=450,y=200)
         
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(im_th))
         
        # Add a PhotoImage to the Canvas
        canvas3.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
def detect():
    global im_th,mask,cv_img,img,clone_img,canvas3,canvas4
    #canvas3.delete("all")
    clone_img = copy.copy(img)
    #cv2.imshow('median filter1',img)
    denoised=cv2.medianBlur(im_th,9)
    #cv2.imshow('median filter',denoised)
    edges = cv2.Canny(denoised,100,200)
    #cv2.imshow('median filter3',edges)
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
    lbl = tk.Label(window, text="Tumor Detection Image",width=20  ,height=2  ,fg="black",font=('times', 10, ' bold ') ) 
    lbl.place(x=660, y=160)
    # Create a canvas that can fit the above image
    canvas4 = tkinter.Canvas(window, width = width, height = height)
    canvas4.place(x=650,y=200)
     
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
     
    # Add a PhotoImage to the Canvas
    canvas4.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.mainloop()
def Feature_extraction():
    global clone_img,contrast,energy,homogeneity,correlation
    im_th=cv2.cvtColor(clone_img, cv2.COLOR_BGR2GRAY)
    g = skimage.feature.greycomatrix(im_th, [1], [0], levels=256, symmetric=False, 
        normed=True)
    contrast = skimage.feature.greycoprops(g, 'contrast')[0][0]
    energy = skimage.feature.greycoprops(g, 'energy')[0][0]
    homogeneity = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
    correlation = skimage.feature.greycoprops(g, 'correlation')[0][0]
    lbl = tk.Label(window, text="CONTRAST:",width=20  ,height=2  ,fg="red" ,font=('times', 10, ' bold ') ) 
    lbl.place(x=850, y=200)
    lbl = tk.Label(window, text="ENERGY:",width=20  ,height=2  ,fg="red"   ,font=('times', 10, ' bold ') ) 
    lbl.place(x=850, y=230)
    lbl = tk.Label(window, text="HOMOGENEITY:",width=20  ,height=2  ,fg="red" ,font=('times', 10, ' bold ') ) 
    lbl.place(x=850, y=260)
    lbl = tk.Label(window, text="CORRELATION:",width=20  ,height=2  ,fg="red" ,font=('times', 10, ' bold ') ) 
    lbl.place(x=850, y=290)
    lbl = tk.Label(window, text="FEATURE EXTRACTION",width=20  ,height=2  ,fg="black" ,font=('times', 15, ' bold ') ) 
    lbl.place(x=900, y=150)
    message = tk.Label(window, text=contrast ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 10, ' bold ')) 
    message.place(x=1000, y=200)
    message = tk.Label(window, text=energy  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 10, ' bold ')) 
    message.place(x=1000, y=230)
    message = tk.Label(window, text=homogeneity   ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 10, ' bold ')) 
    message.place(x=1000, y=260)
    message = tk.Label(window, text=correlation   ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 10, ' bold ')) 
    message.place(x=1000, y=290)
    print(contrast)
    print(energy)
    print(homogeneity)
    print(correlation)
    window.mainloop()
def Train_data():
    datadir = "F:\\2019-2020\\BE\\Register\\2.SGI Brain Tumor Detection Using Machine Learning\\Phase6\\Brain Toumer"
    Categories = ["no","yes"]
    j=0
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    lbl = tk.Label(window, text="Training",width=20  ,height=2  ,fg="red" ,font=('times', 13, ' bold ') ) 
    lbl.place(x=850, y=355)
    progress = Progressbar(window, orient = HORIZONTAL, 
              length = 500, mode = 'determinate',maximum =253) 
    for category in Categories:
        path = os.path.join(datadir,category)
        #print(len(os.listdir(path))
        for img in os.listdir(path):        
            #name= filedialog.askopenfilename() 
            img = cv2.imread(os.path.join(path,img))
            dim=(180,218)
            img = cv2.resize(img,dim)
            img1=img
            #print(img.shape[:2])
            im_out=np.zeros(img.shape[:2], dtype="uint8") * 255
            mask = np.zeros(img.shape[:2], dtype="uint8") * 255
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width= img_gray.shape
            ret,im_th= cv2.threshold(img_gray,180,255,cv2.THRESH_BINARY)
            #cv2.imshow('image Throshold', im_th)
            denoised=cv2.medianBlur(im_th,9)
            #cv2.imshow('image Denoised', denoised)
            edges = cv2.Canny(denoised,100,200)
            #cv2.imshow('Edges',edges)
                # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(denoised,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
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
            markers = cv2.watershed(img,markers)
            img[markers == 1] = [255,255,255]
            img1[markers == 1] = [0]
            im_th=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('image marker',img)
            img1 = cv2.bitwise_and(img, img, mask=mask)
            g = skimage.feature.greycomatrix(im_th, [1], [0], levels=256, symmetric=False, 
            normed=True)
            contrast = skimage.feature.greycoprops(g, 'contrast')[0][0]
            energy = skimage.feature.greycoprops(g, 'energy')[0][0]
            homogeneity = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
            correlation = skimage.feature.greycoprops(g, 'correlation')[0][0]
            sheet1.write(j, 0, contrast) 
            sheet1.write(j, 1, energy) 
            sheet1.write(j, 2, homogeneity) 
            sheet1.write(j, 3, correlation)
            print(j)
            j=j+1
            wb.save('xlwt example.xls')

  
            progress['value'] = j
            window.update_idletasks() 
            progress.place(x=850,y=400)
    lbl = tk.Label(window, text="Training Completed",width=20  ,height=2  ,fg="red" ,font=('times', 13, ' bold ') ) 
    lbl.place(x=850, y=355)
        #print(pandas.read_excel('xlwt example.xls'))
def classifier():
    global contrast,energy,homogeneity,correlation
    df = pd.read_csv("xlwt example.csv")
    y=df.out
    x=df.drop('out',axis=1)
    print(x)
    print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 1)
    clf=MLPClassifier()
    model=clf.fit(x_train,y_train)
    predication=model.predict([[contrast,energy,homogeneity,correlation]])
    print(predication)
    lbl = tk.Label(window, text="Result:",width=20  ,height=2  ,fg="black" ,font=('times', 13, ' bold ') ) 
    lbl.place(x=850, y=420)
    message = tk.Label(window, text=predication ,fg="red"  ,width=30  ,height=2 ,font=('times', 13, ' bold ')) 
    message.place(x=980, y=420)

canvas1 = tkinter.Canvas(window, width = 300, height = 400)
canvas1.place(x=50,y=200)
canvas2 = tkinter.Canvas(window, width = 300, height = 400)
canvas2.place(x=250,y=200)
canvas3 = tkinter.Canvas(window, width = 300, height = 400)
canvas3.place(x=450,y=200)
canvas4 = tkinter.Canvas(window, width = 300, height = 400)
canvas4.place(x=50,y=200)

btn1=Button(text='Load Image', command=answer,width=10  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn1.place(x=50,y=500)
btn2=Button(text='Rgb2gary', command=rgb2gray,width=10  ,height=2 ,activebackground = "Green" ,font=('times', 15, ' bold '))
btn2.place(x=200,y=500)
btn3=Button(text='Gray Thr', command=Thresholding,width=10  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn3.place(x=350,y=500)
btn4=Button(text='find Tumor', command=detect,width=10  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn4.place(x=500,y=500)
btn5=Button(text='Feature Extraction',command=Feature_extraction,width=15  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn5.place(x=650,y=500)
btn6=Button(text='Train Dataset',command=Train_data,width=15  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn6.place(x=860,y=500)
btn7=Button(text='Classifiaction',command=classifier,width=15  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
btn7.place(x=1070,y=500)
# Load an image using OpenCV

