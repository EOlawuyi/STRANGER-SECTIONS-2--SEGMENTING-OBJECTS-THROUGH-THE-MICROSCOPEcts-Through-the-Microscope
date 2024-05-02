from PIL import Image as im
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import *
from tkinter import ttk
import sys
import numpy as np
import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import skimage as ski
import skimage.feature
import pandas as pd
import scipy.stats as stats
from scipy.stats import entropy
from skimage import feature, measure
from skimage.measure import label, regionprops, regionprops_table
import pyarrow.parquet as pa
import string
from openpyxl import load_workbook
from openpyxl import Workbook
import openpyxl
import csv
import cv2
import json
from statistics import mode
from statistics import mean
from skimage.io import imread, imshow

#This is the code that belongs to Olorogun Engineer Enoch O. Ejofodomi in his Collaboration with Shell Onward.
#This code also belongs to Engineer Francis Olawuyi in his collaboration with Shell Onward.
#The code also belongs to the following people
#1. GODSWILL OFUALAGBA C.E.O. SWILLAS ENERGY LIMITED.
#2. DR. MICHAEL OLAWUYI
#3. DR. DAMILOLA SUNDAY OLAWUYI
#4. ENGINEER DEBORAH OLAWUYI
#5. ENGINEER JOSHUA OLAWUYI
#6. ENGINEER JOSEPH OLAWUYI
#7. ENGINEER ONOME EJOFODOMI
#8. ENGINEER EFEJERA EJOFODOMI
#9. ENGINEER FRANCIS OLAWUYI
#10. DR. MATTHEW OLAWUYI
#11. ENGINEER ENOCH O. EJOFODOMI
#12. OCHAMUKE EJOFODOMI
#13. ENGINEER ONOME OMIYI
#14. MS. KOME ODU
#15. MR. KAYODE ADEDIPE
#16. MR. OMAFUME EJOFODOMI
#17. MR. NICHOLAS MONYENYE
#18. ENGINEER AYO ADEGITE
#19. ENGINEER ESOSA EHANIRE
#20. Ms. NANAYE INOKOBA
#21. Ms. YINKA OLAREWAJU-ALO
#22. Ms. ERKINAY ABLIZ
#23. Ms. FAEZEH RAZOUYAN
#24. MRS. TEVUKE EJOFODOMI
#25. MR.ONORIODE AGGREH
#26. MS. NDIDI IKEMEFUNA
#27. MS. ENAJITE AGGREH
#28. DR. ESTHER OLAWUYI
#29  MS. ISI OMIYI
#30. DR. JASON ZARA
#31. DR. VESNA ZDERIC
#32. DR. AHMED JENDOUBI
#33. DR. MOHAMMED CHOUIKHA
#34. MS. SHANI ROSS


# APRIL 19, 2024.

mainwindow = tk.Tk()
mainwindow.geometry("1400x700")

            
#Load 25 Test Images
Imagefile = ["2fpvuk.jpg", "2otd5q.jpg", "4uhzc7.jpg", "4ywofb.jpg", "cgjz2a.jpg", "gp0mak.jpg", "jbpvyh.jpg", "n7ozhj.jpg", "ont2xr.jpg", "oqr1h3.jpg", "v0t9rk.jpg", "v7dlpt.jpg", "vktqud.jpg", "vlridu.jpg", "vyo284.jpg", "w7v4b5.jpg", "whvcmt.jpg", "widroh.jpg", "wocbyu.jpg", "x4vowt.jpg", "xokfeh.jpg", "y0hsj8.jpg", "y3b47k.jpg", "yir071.jpg", "zyvdo0.jpg" ]
         
#Assign names to Store Result Files as Dictated in the Submission File
Imagefile3 = ["2fpvuk_pred.npy", "2otd5q_pred.npy", "4uhzc7_pred.npy",
              "4ywofb_pred.npy", "cgjz2a_pred.npy", "gp0mak_pred.npy",
              "jbpvyh_pred.npy", "n7ozhj_pred.npy", "ont2xr_pred.npy",
              "oqr1h3_pred.npy", "v0t9rk_pred.npy", "v7dlpt_pred.npy",
              "vktqud_pred.npy", "vlridu_pred.npy", "vyo284_pred.npy",
              "w7v4b5_pred.npy", "whvcmt_pred.npy", "widroh_pred.npy",
              "wocbyu_pred.npy", "x4vowt_pred.npy", "xokfeh_pred.npy",
              "y0hsj8_pred.npy", "y3b47k_pred.npy", "yir071_pred.npy",
              "zyvdo0_pred.npy"
             ]

#Analyze Each of the 25 Test Images
for z in range(0,25):
   #Read in Image
   datab = cv2.imread(Imagefile[z])
   datac = cv2.imread(Imagefile[z])
   datad = cv2.imread(Imagefile[z])
   datae = cv2.imread(Imagefile[z])
   dataf = cv2.imread(Imagefile[z])
   datag = cv2.imread(Imagefile[z])
   datah = cv2.imread(Imagefile[z])
   datai = cv2.imread(Imagefile[z])
   dataj = cv2.imread(Imagefile[z])
   datak = cv2.imread(Imagefile[z])
   datal = cv2.imread(Imagefile[z])
   datam = cv2.imread(Imagefile[z])
   datan = cv2.imread(Imagefile[z])
   datao = cv2.imread(Imagefile[z])
   datap = cv2.imread(Imagefile[z])
   dataq = cv2.imread(Imagefile[z])
   datar = cv2.imread(Imagefile[z])
   datas = cv2.imread(Imagefile[z])
   datat = cv2.imread(Imagefile[z])
   datau = cv2.imread(Imagefile[z])
   datav = cv2.imread(Imagefile[z])
   dataw = cv2.imread(Imagefile[z])
   datax = cv2.imread(Imagefile[z])
   datay = cv2.imread(Imagefile[z])
   dataz = cv2.imread(Imagefile[z])
   datafff = cv2.imread(Imagefile[z])
   gray = cv2.cvtColor(datab, cv2.COLOR_BGR2GRAY)
   print(datab)
   print(datab.size)
   print(datab.shape)
   img = im.fromarray(datab, 'RGB')



   #Color Thresholding for Macerals (Liptinite - Dark Gray/brown,
   # (Vitrinite - Medium to Light Gray) & (Inertinite - White)

   #Color Thresholding for Liptinite - Dark Brown,
   [a1,b1,c1] = datab.shape
   print(a1)
   print(b1)
   print(c1)
   for i in range(0,a1):
      for j in range(0,b1):
         if((datae[i,j,0] > 60) & (datae[i,j,0] < 255)):
            if((datae[i,j,1] > 4) & (datae[i,j,1] < 180)):
               if( ((datae[i,j,2] > 100) & (datae[i,j,2] < 180))):# | ((datab[i,j,2] > 40) & (datab[i,j,2] < 60)) ):
             #print("Got in")
                datad[i,j,0] = 255
                datad[i,j,1] = 0
                datad[i,j,2] = 0

   data1 = dataf
   for i in range(0,a1):
      for j in range(0,b1):
           if((datad[i,j,0] != 255) & (datad[i,j,1] != 0) & (datad[i,j,2] != 0) ):
              data1[i,j,0] = 0
              data1[i,j,1] = 0
              data1[i,j,2] = 0


   #Futher Winnowing 1
   # target color code: [131 121 121]
   data2 = datag
   for i in range(0,a1):
      for j in range(0,b1):
         if((datae[i,j,0] > 111) & (datae[i,j,0] < 151)):
            if((datae[i,j,1] > 101) & (datae[i,j,1] < 141)):
               if( ((datae[i,j,2] > 101) & (datae[i,j,2] < 141))):# | ((datab[i,j,2] > 40) & (datab[i,j,2] < 60)) ):
             #print("Got in")
                data2[i,j,0] = 255
                data2[i,j,1] = 0
                data2[i,j,2] = 0


   data3 = datag
   for i in range(0,a1):
      for j in range(0,b1):
           if((data2[i,j,0] != 255) & (data2[i,j,1] != 0) & (data2[i,j,2] != 0) ):
              data3[i,j,0] = 0
              data3[i,j,1] = 0
              data3[i,j,2] = 0


   #Futher Winnowing 2
   # target oolor code: [175 161 155]
   data4 = datah
   for i in range(0,a1):
      for j in range(0,b1):
         if((datae[i,j,0] > 145) & (datae[i,j,0] < 205)):
            if((datae[i,j,1] > 131) & (datae[i,j,1] < 191)):
               if( ((datae[i,j,2] > 125) & (datae[i,j,2] < 185))):# | ((datab[i,j,2] > 40) & (datab[i,j,2] < 60)) ):
             #print("Got in")
                data4[i,j,0] = 255
                data4[i,j,1] = 0
                data4[i,j,2] = 0


   data5 = datah
   for i in range(0,a1):
      for j in range(0,b1):
           if((data4[i,j,0] != 255) & (data4[i,j,1] != 0) & (data4[i,j,2] != 0) ):
              data5[i,j,0] = 0
              data5[i,j,1] = 0
              data5[i,j,2] = 0

             
   img3 = im.fromarray(data1, 'RGB')
   img4 = im.fromarray(data3, 'RGB')
   img5 = im.fromarray(data5, 'RGB')



   #Converge Liptinite Thresholding (Dark Brown) Color Winnowing 1 & 2 into
   # ONE Single Image
   data6 = datai
   for i in range(0,a1):
      for j in range(0,b1):
         if(((data3[i,j,0] == 0) & (data3[i,j,1] == 0) & (data3[i,j,2] == 0)) & ((data5[i,j,0] == 0) & (data5[i,j,1] == 0) & (data5[i,j,2] == 0) )):
              data6[i,j,0] = 0
              data6[i,j,1] = 0
              data6[i,j,2] = 0

   img7 = im.fromarray(data6, 'RGB')


   graytest = cv2.cvtColor(data6, cv2.COLOR_BGR2GRAY)
   #Grayscale Thresholding to extract lines in Test Image
   [d,e] = graytest.shape
   graytest2 = cv2.cvtColor(data6, cv2.COLOR_BGR2GRAY)
   #perform Region Props on Thresholded Test Image
   lineimagecctest = np.array(graytest)
   #Select Pixels Greater than 100 with a mask
   masktest = lineimagecctest > 100
   labelstest = measure.label(masktest)

   #Segment out Regions
   regionstest = measure.regionprops(labelstest, lineimagecctest)
   numlabelstest = len(regionstest)
   regionstest = regionprops_table(labelstest, properties=('area', 'coords'))
   #regionstest = regionprops_table(labelstest, properties=('area', 'perimeter'))
   #print(regions)
   pd.DataFrame(regionstest)
   y = pd.DataFrame(regionstest)
   #Get Shape and Size of Regions
   [a1,b1] = y.shape

   #Select Only Regions Greater than 500 Pixels and Get their Line Count
   linecounttest = 0
   #Array Variable to hold Number of Lines Detected
   TotalRegions = 0
   graytest3 = np.zeros((e,d))
   gray1 = np.zeros((7000,1))
   gray1b = np.zeros((7000,1))
   gray2 = np.zeros((7000,2))
   threshold = np.zeros((d,e))

   for j in range(0,a1):
      if(y.values[j,0] > 1000):
         gray1[linecounttest,0] = y.values[j,0]
         linecounttest = linecounttest + 1
         gray1b[linecounttest] = j

    
   print("linecounttest")
   print(linecounttest)


   img8 = im.fromarray(np.uint8(graytest3 * 255), 'L')
   for m in range(0,linecounttest):
      [size1, size2] = y.values[int(gray1b[m]),1].shape
      print("starting")
      print(size1)
      print(size2)
      for p in range(0, size1-1):
         img8.putpixel((int(y.values[int(gray1b[m]),1][p,0]),int(y.values[int(gray1b[m]),1][p,1])), 255)        

   #Final Liptinite Dark Brown Image Detected is img8
   print("Final Image for Liptinite - Dark Brown:")

   #Transposing to right size. Transpose and then Rotate to see Final Image
   img9 = img8.transpose(1)
   img10 = img9.rotate(-90)

  #Final Liptinite Dark Brown Image Detected is Image LiptiniteBrown
   LiptiniteBrown = img10


   print("Starting Liptinite Dark Gray Image Processing")
   #Color Thresholding for Liptinite - Dark Gray,
   # Ideal Values: [149 151 145]
   datam = cv2.imread(Imagefile[z])
   [a1,b1,c1] = datab.shape
   print(a1)
   print(b1)
   print(c1)

   for i in range(0,a1):
      for j in range(0,b1):
         if((datam[i,j,0] > 148) & (datam[i,j,0] < 150)):
            if((datam[i,j,1] > 150) & (datam[i,j,1] < 152)):
               if( ((datam[i,j,2] > 144) & (datam[i,j,2] <146))):# | ((datab[i,j,2] > 40) & (datab[i,j,2] < 60)) ):
                datad[i,j,0] = 255
                datad[i,j,1] = 0
                datad[i,j,2] = 0

   data1 = dataf
   for i in range(0,a1):
      for j in range(0,b1):
           if((datad[i,j,0] != 255) & (datad[i,j,1] != 0) & (datad[i,j,2] != 0) ):
              data1[i,j,0] = 0
              data1[i,j,1] = 0
              data1[i,j,2] = 0


   print("Final Liptrinite Dark Gray Image 1")           
   img3 = im.fromarray(data1, 'RGB')

   graytest = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #Grayscale Thresholding to extract lines in Test Image
   [d,e] = graytest.shape
   graytest2 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #perform Region Props on Thresholded Test Image
   lineimagecctest = np.array(graytest)
   #Select Pixels Greater than 100 with a mask
   masktest = lineimagecctest > 100
   labelstest = measure.label(masktest)

   #Segment out Regions
   regionstest = measure.regionprops(labelstest, lineimagecctest)
   numlabelstest = len(regionstest)
   regionstest = regionprops_table(labelstest, properties=('area', 'coords'))
   #regionstest = regionprops_table(labelstest, properties=('area', 'perimeter'))
   #print(regions)
   pd.DataFrame(regionstest)
   y = pd.DataFrame(regionstest)
   #Get Shape and Size of Regions
   [a1,b1] = y.shape

   #Select Only Regions Greater than 500 Pixels and Get their Line Count
   linecounttest = 0
   #Array Variable to hold Number of Lines Detected
   TotalRegions = 0
   graytest3 = np.zeros((e,d))
   gray1 = np.zeros((7000,1))
   gray1b = np.zeros((7000,1))
   gray2 = np.zeros((7000,2))
   threshold = np.zeros((d,e))


   for j in range(0,a1):
      #print(y.values[j,0])
      if(y.values[j,0] > 1000):
         gray1[linecounttest,0] = y.values[j,0]
         linecounttest = linecounttest + 1
         gray1b[linecounttest] = j
    
   print("linecounttest for 2")
   print(linecounttest)


   img8 = im.fromarray(np.uint8(graytest3 * 255), 'L')
   for m in range(0,linecounttest):
      [size1, size2] = y.values[int(gray1b[m]),1].shape
      print("starting")
      print(size1)
      print(size2)
      for p in range(0, size1-1):
         img8.putpixel((int(y.values[int(gray1b[m]),1][p,0]),int(y.values[int(gray1b[m]),1][p,1])), 255)

   print("Final image 2 for Liptinite Dark Gray Image:")


   #Transposing to right size. Transpose and then Rotate to see Final Image
   img9 = img8.transpose(1)
   img10 = img9.rotate(-90)

   #Final Liptinite Dark Gray Detection Image is Image LiptiniteDarkGray
   LiptiniteDarkGray = img10


   #Do the same thing for the other macerels:
   # (Vitrinite - Medium to Light Gray) & (Inertinite - White)
   #Next: Vitrinite - Medium to Light Gray 
   #Color Thresholding for Vitrinite - Medium to Light Gray
   # Target Color Range: [125 123 135]

   print("Starting Vitrinite Medium to Light Gray Processing")
   #Color Thresholding for Vitrinite - Medium to Light Gray,
   # Ideal Values: [125 123 135]
   datan = cv2.imread(Imagefile[z])
   [a1,b1,c1] = datab.shape
   print(a1)
   print(b1)
   print(c1)
   for i in range(0,a1):
      for j in range(0,b1):
         if((datan[i,j,0] > 124) & (datan[i,j,0] < 126)):
            if((datan[i,j,1] > 122) & (datan[i,j,1] < 124)):
               if( ((datan[i,j,2] > 134) & (datan[i,j,2] <136))):# | ((datab[i,j,2] > 40) & (datab[i,j,2] < 60)) ):
                datad[i,j,0] = 255
                datad[i,j,1] = 0
                datad[i,j,2] = 0

   data1 = dataf
   for i in range(0,a1):
      for j in range(0,b1):
           if((datad[i,j,0] != 255) & (datad[i,j,1] != 0) & (datad[i,j,2] != 0) ):
              data1[i,j,0] = 0
              data1[i,j,1] = 0
              data1[i,j,2] = 0


   print("Final Vitrinite Medium to Light Gray Image 1")           
   img3 = im.fromarray(data1, 'RGB')
   img3.show()

   graytest = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #Grayscale Thresholding to extract lines in Test Image
   [d,e] = graytest.shape
   graytest2 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #perform Region Props on Thresholded Test Image
   lineimagecctest = np.array(graytest)
   #Select Pixels Greater than 100 with a mask
   masktest = lineimagecctest > 100
   labelstest = measure.label(masktest)

   #Segment out Regions
   regionstest = measure.regionprops(labelstest, lineimagecctest)
   numlabelstest = len(regionstest)
   regionstest = regionprops_table(labelstest, properties=('area', 'coords'))
   #regionstest = regionprops_table(labelstest, properties=('area', 'perimeter'))
   #print(regions)
   pd.DataFrame(regionstest)
   y = pd.DataFrame(regionstest)
   #Get Shape and Size of Regions
   [a1,b1] = y.shape

   #Select Only Regions Greater than 500 Pixels and Get their Line Count
   linecounttest = 0
   #Array Variable to hold Number of Lines Detected
   TotalRegions = 0
   graytest3 = np.zeros((e,d))
   gray1 = np.zeros((7000,1))
   gray1b = np.zeros((7000,1))
   gray2 = np.zeros((7000,2))
   threshold = np.zeros((d,e))

   one = 0
   two = 0

   for j in range(0,a1):
      if(y.values[j,0] > 1000):
         gray1[linecounttest,0] = y.values[j,0]
         linecounttest = linecounttest + 1
         gray1b[linecounttest] = j
    
   print("linecounttest for 2")
   print(linecounttest)


   img8 = im.fromarray(np.uint8(graytest3 * 255), 'L')
   for m in range(0,linecounttest):
      [size1, size2] = y.values[int(gray1b[m]),1].shape
      print("starting")
      print(size1)
      print(size2)
      for p in range(0, size1-1):
         img8.putpixel((int(y.values[int(gray1b[m]),1][p,0]),int(y.values[int(gray1b[m]),1][p,1])), 255)

   print("Final Image for Vitrinite Medium to Light Gray Processing 2:")


   #Transposing to right size. Transpose and then Rotate to see Final Image
   img9 = img8.transpose(1)
   img10 = img9.rotate(-90)

   #Final Vitrinite Medium to Light Gray Image Detected is Image VitriniteMediumtoLightGray
   VitriniteMediumtoLightGray = img10



   #Do the same thing for the other macerels:
   # Inertinite - White
   #Next: Inertinite - White 
   #Color Thresholding for Inertiinite - White
   # Target Color Range: [194 200 229]

   print("Starting Inertinite White Processing")
   #Color Thresholding for Inertinite - White
   # Ideal Values: [194 200 229]
   datao = cv2.imread(Imagefile[z])
   [a1,b1,c1] = datab.shape
   print(a1)
   print(b1)
   print(c1)
   for i in range(0,a1):
      for j in range(0,b1):
         if((datao[i,j,0] > 189) & (datao[i,j,0] < 199)):
            if((datao[i,j,1] > 195) & (datao[i,j,1] < 205)):
               if( ((datao[i,j,2] > 224) & (datao[i,j,2] <234))):
                datad[i,j,0] = 255
                datad[i,j,1] = 0
                datad[i,j,2] = 0

   data1 = dataf
   for i in range(0,a1):
      for j in range(0,b1):
           if((datad[i,j,0] != 255) & (datad[i,j,1] != 0) & (datad[i,j,2] != 0) ):
              data1[i,j,0] = 0
              data1[i,j,1] = 0
              data1[i,j,2] = 0


   print("Final Inertinite White Image 1")           
   img3 = im.fromarray(data1, 'RGB')

   graytest = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #Grayscale Thresholding to extract lines in Test Image
   [d,e] = graytest.shape
   graytest2 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #perform Region Props on Thresholded Test Image
   lineimagecctest = np.array(graytest)
   #Select Pixels Greater than 100 with a mask
   masktest = lineimagecctest > 100
   labelstest = measure.label(masktest)

   #Segment out Regions
   regionstest = measure.regionprops(labelstest, lineimagecctest)
   numlabelstest = len(regionstest)
   regionstest = regionprops_table(labelstest, properties=('area', 'coords'))
   #regionstest = regionprops_table(labelstest, properties=('area', 'perimeter'))
   #print(regions)
   pd.DataFrame(regionstest)
   y = pd.DataFrame(regionstest)
   #Get Shape and Size of Regions
   [a1,b1] = y.shape

   #Select Only Regions Greater than 500 Pixels and Get their Line Count
   linecounttest = 0
   #Array Variable to hold Number of Lines Detected
   TotalRegions = 0
   graytest3 = np.zeros((e,d))
   gray1 = np.zeros((7000,1))
   gray1b = np.zeros((7000,1))
   gray2 = np.zeros((7000,2))
   threshold = np.zeros((d,e))

   for j in range(0,a1):
      #print(y.values[j,0])
      if(y.values[j,0] > 1000):
         gray1[linecounttest,0] = y.values[j,0]
         linecounttest = linecounttest + 1
         gray1b[linecounttest] = j

    
   print("linecounttest for 2")
   print(linecounttest)


   img8 = im.fromarray(np.uint8(graytest3 * 255), 'L')
   for m in range(0,linecounttest):
      [size1, size2] = y.values[int(gray1b[m]),1].shape
      print("starting")
      print(size1)
      print(size2)
      for p in range(0, size1-1):
         img8.putpixel((int(y.values[int(gray1b[m]),1][p,0]),int(y.values[int(gray1b[m]),1][p,1])), 255)


   print("Final Image for Inertinite White Image 2:")


   #Transposing to right size. Transpose and then Rotate to see Final Image
   img9 = img8.transpose(1)
   img10 = img9.rotate(-90)

   #Final Inertinite White Detected Image is Image InertiniteWhite
   InertiniteWhite = img10


   #Do the same thing for the other macerels:
   #Inertinite - Blue
   #Next: Inertinite - Blue 
   #Color Thresholding for Inertiinite - Blue
   #Target Color Range: [49 255 255]


   print("Starting Inertinite Blue Processing")
   #Color Thresholding for Inertinite - Blue
   # Ideal Values: [49 255 255]
   datap = cv2.imread(Imagefile[z])
   [a1,b1,c1] = datab.shape
   print(a1)
   print(b1)
   print(c1)
   for i in range(0,a1):
      for j in range(0,b1):
         if((datap[i,j,0] > 44) & (datap[i,j,0] < 54)):
            if((datap[i,j,1] > 250)): 
               if( ((datap[i,j,2] > 250))):
                   datad[i,j,0] = 255
                   datad[i,j,1] = 0
                   datad[i,j,2] = 0

   data1 = dataf
   for i in range(0,a1):
      for j in range(0,b1):
           if((datad[i,j,0] != 255) & (datad[i,j,1] != 0) & (datad[i,j,2] != 0) ):
              data1[i,j,0] = 0
              data1[i,j,1] = 0
              data1[i,j,2] = 0


   print("Final Inertinite Blue Image 1")           
   img3 = im.fromarray(data1, 'RGB')
   img3.show()

   graytest = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #Grayscale Thresholding to extract lines in Test Image
   [d,e] = graytest.shape
   graytest2 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
   #perform Region Props on Thresholded Test Image
   lineimagecctest = np.array(graytest)
   #Select Pixels Greater than 100 with a mask
   masktest = lineimagecctest > 100
   labelstest = measure.label(masktest)

   #Segment out Regions
   regionstest = measure.regionprops(labelstest, lineimagecctest)
   numlabelstest = len(regionstest)
   regionstest = regionprops_table(labelstest, properties=('area', 'coords'))
   #regionstest = regionprops_table(labelstest, properties=('area', 'perimeter'))
   #print(regions)
   pd.DataFrame(regionstest)
   y = pd.DataFrame(regionstest)
   #Get Shape and Size of Regions
   [a1,b1] = y.shape

   #Select Only Regions Greater than 500 Pixels and Get their Line Count
   linecounttest = 0
   #Array Variable to hold Number of Lines Detected
   TotalRegions = 0
   graytest3 = np.zeros((e,d))
   gray1 = np.zeros((7000,1))
   gray1b = np.zeros((7000,1))
   gray2 = np.zeros((7000,2))
   threshold = np.zeros((d,e))

   for j in range(0,a1):
      #print(y.values[j,0])
      if(y.values[j,0] > 1000):
         gray1[linecounttest,0] = y.values[j,0]
         linecounttest = linecounttest + 1
         gray1b[linecounttest] = j
    
   print("linecounttest for 2")
   print(linecounttest)


   img8 = im.fromarray(np.uint8(graytest3 * 255), 'L')
   img8.show()
   for m in range(0,linecounttest):
      [size1, size2] = y.values[int(gray1b[m]),1].shape
      print("starting")
      print(size1)
      print(size2)
      for p in range(0, size1-1):
         img8.putpixel((int(y.values[int(gray1b[m]),1][p,0]),int(y.values[int(gray1b[m]),1][p,1])), 255)

   print("Final Image for Inertinite Blue Image 2:")

   #Final InertiniteBlue Detected Image is Image InertiniteBlue
   InertiniteBlue = img10

   #Congregate ALL FIVE (5) IMAGES from Macerel Processing into ONE (1) SINGLE Image
   print("Getting FINAL IMAGE from Marcerel Detection Algorithm")
   dataq = cv2.imread(Imagefile[z])
   [a1,b1,c1] = datab.shape
   FinalImageMacerel = np.zeros((d,e))
   FinalImageMacerel2 = img10
   print(a1)
   print(b1)
   print(c1)
   for i in range(0,a1):
      for j in range(0,b1):
         if((LiptiniteBrown.getpixel((i,j)) != 0) | (LiptiniteDarkGray.getpixel((i,j)) != 0) | (VitriniteMediumtoLightGray.getpixel((i,j)) != 0) | (InertiniteWhite.getpixel((i,j)) != 0) | (InertiniteBlue.getpixel((i,j)) != 0)):
            FinalImageMacerel2.putpixel((i,j), 255)
            FinalImageMacerel[i,j] = 2
         else:
            FinalImageMacerel2.putpixel((i,j), 0)
            
   print("Final Macerel Image 2:")           
   #FinalImageMarcerel2 is the Final Image from the Detection Algorithm
   FinalImageMacerel2.show()

   #We store the final Image from the Detection Algorithm as the Required .NPY File for Submission
   np.save(Imagefile3[z],FinalImageMacerel)



