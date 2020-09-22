import xlwt 
from xlwt import Workbook
import os
import pandas  as pd

def Train_data():
    datadir = "H:\RITESH\Python\ML\Phase3\Brain Toumer"
    Categories = ["no","yes"]
    j=0
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    for category in Categories:
        path = os.path.join(datadir,category)
        #print(len(os.listdir(path))
        for img in os.listdir(path):        
            #name= filedialog.askopenfilename() 
            img = cv2.imread(os.path.join(path,img))
            img1=img
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
        #print(pandas.read_excel('xlwt example.xls'))
