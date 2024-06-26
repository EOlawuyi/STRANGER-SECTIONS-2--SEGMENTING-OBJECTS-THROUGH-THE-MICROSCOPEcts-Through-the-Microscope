from PIL import Image as im
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import *
from tkinter import ttk
import sys
import numpy as np
import imageio
#import imageio.v3 as iio
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

            
#Load 362 Images
Imagefile = ["2otd5q.jpg",  "2otd5q.jpg", "4uhzc7.jpg",
              "4ywofb.jpg", "cgjz2a.jpg", "gp0mak.jpg",
              "jbpvyh.jpg", "n7ozhj.jpg", "ont2xr.jpg",
              "oqr1h3.jpg", "v0t9rk.jpg", "v7dlpt.jpg",
              "vktqud.jpg", "vlridu.jpg", "vyo284.jpg",
              "w7v4b5.jpg", "whvcmt.jpg", "widroh.jpg",
              "wocbyu.jpg", "x4vowt.jpg", "xokfeh.jpg",
              "y0hsj8.jpg", "y3b47k.jpg", "yir071.jpg",
              "zyvdo0.jpg"
             ]

         
#Imagefile2 = ["2otd5q_gt.npy", "2otd5q_pred.npy", "4uhzc7_pred.npy",
#              "4ywofb_pred.npy", "cgjz2a_pred.npy", "gp0mak_pred.npy",
#              "jbpvyh_pred.npy", "n7ozhj_pred.npy", "ont2xr_pred.npy",
#              "oqr1h3_pred.npy", "v0t9rk_pred.npy", "v7dlpt_pred.npy",
#              "vktqud_pred.npy", "vlridu_pred.npy", "vyo284_pred.npy",
#              "w7v4b5_pred.npy", "whvcmt_pred.npy", "widroh_pred.npy",
#              "wocbyu_pred.npy", "x4vowt_pred.npy", "xokfeh_pred.npy",
#              "y0hsj8_pred.npy", "y3b47k_pred.npy", "yir071_pred.npy",
#              "zyvdo0_pred.npy"
#             ]

Imagefile3 = ["2otd5q_gt.npy", "2otd5q_pred.npy", "4uhzc7_pred.npy",
              "4ywofb_pred.npy", "cgjz2a_pred.npy", "gp0mak_pred.npy",
              "jbpvyh_pred.npy", "n7ozhj_pred.npy", "ont2xr_pred.npy",
              "oqr1h3_pred.npy", "v0t9rk_pred.npy", "v7dlpt_pred.npy",
              "vktqud_pred.npy", "vlridu_pred.npy", "vyo284_pred.npy",
              "w7v4b5_pred.npy", "whvcmt_pred.npy", "widroh_pred.npy",
              "wocbyu_pred.npy", "x4vowt_pred.npy", "xokfeh_pred.npy",
              "y0hsj8_pred.npy", "y3b47k_pred.npy", "yir071_pred.npy",
              "zyvdo0_pred.npy"
             ]
              
for z in range(0,1):
   #Read in Image
   #datab = cv2.imread("jbpvyh.jpg")
   #datab = cv2.imread("yir071.jpg", cv2.IMREAD_UNCHANGED)
   # datab = imageio.v2.imread("w7v4b5.jpg")
   datab = imageio.v2.imread(Imagefile[z])
   img = im.fromarray(datab, mode=None)
   #img = im.fromarray(datab, 'RGB')
  #img.show()

   #Blurs
   datab1 = cv2.medianBlur(datab, 9)  #chosen blur
   #datab2 = cv2.GaussianBlur(datab, (3,3), 0)
   #datab3 = cv2.blur(datab,(5,5))
   #datab4 = cv2.medianBlur(datab,9)
   #datab5 = cv2.bilateralFilter(datab, 9, 75, 75)
   #kernel = np.ones((5,5),np.float32)/25 #blur
   #databbb = cv2.filter2D(databb, -3, kernel)

   #Show Blurred Image
   imgg2 = im.fromarray(datab1, mode=None)
   #imgg2.show()

   #Sharpen Image
   kernel2 = np.array([[0, -1, 0], [-1, 5, -1],[0, -1, 0]]) #sharpen
   databb = cv2.filter2D(datab1, -3, kernel2) #inner value could be -1, -2, or -3

   #Show Sharpened Image
   imgg3 = im.fromarray(databb, mode=None)
   imgg3.show()   
   
   datac = imageio.imread(Imagefile[z])
   datad = imageio.imread(Imagefile[z])
   datae = imageio.imread(Imagefile[z])
   dataf = imageio.imread(Imagefile[z])
   datag = imageio.imread(Imagefile[z])
   datah = imageio.imread(Imagefile[z])
   datai = imageio.imread(Imagefile[z])
   dataj = imageio.imread(Imagefile[z])
   datak = imageio.imread(Imagefile[z])
   datal = imageio.imread(Imagefile[z])
   datam = imageio.imread(Imagefile[z])
   datan = imageio.imread(Imagefile[z])
   datao = imageio.imread(Imagefile[z])
   datap = imageio.imread(Imagefile[z])
   dataq = imageio.imread(Imagefile[z])
   datar = imageio.imread(Imagefile[z])
   datas = imageio.imread(Imagefile[z])
   datat = imageio.imread(Imagefile[z])
   datau = imageio.imread(Imagefile[z])
   datav = imageio.imread(Imagefile[z])
   dataw = imageio.imread(Imagefile[z])
   datax = imageio.imread(Imagefile[z])
   datay = imageio.imread(Imagefile[z])
   dataz = imageio.imread(Imagefile[z])
   datafff = imageio.imread(Imagefile[z])



#   datac = imageio.imread("w7v4b5.jpg")
#   datad = imageio.imread("w7v4b5.jpg")
#   datae = imageio.imread("w7v4b5.jpg")
#   img2 = im.fromarray(datae, mode=None)
 #  img = im.fromarray(datab, 'RGB')
#   img2.show()
   
#   dataf = imageio.imread("w7v4b5.jpg")
#   datag = imageio.imread("w7v4b5.jpg")
#   datah = imageio.imread("w7v4b5.jpg")
#   datai = imageio.imread("w7v4b5.jpg")
#   dataj = imageio.imread("w7v4b5.jpg")
#   datak = imageio.imread("w7v4b5.jpg")
#   datal = imageio.imread("w7v4b5.jpg")
#   datam = imageio.imread("w7v4b5.jpg")
#   datan = imageio.imread("w7v4b5.jpg")
#   datao = imageio.imread("w7v4b5.jpg")
#   datap = imageio.imread("w7v4b5.jpg")
#   dataq = imageio.imread("w7v4b5.jpg")
#   datar = imageio.imread("w7v4b5.jpg")
#   datas = imageio.imread("w7v4b5.jpg")
#   datat = imageio.imread("w7v4b5.jpg")
#   datau = imageio.imread("w7v4b5.jpg")
#  datav = imageio.imread("w7v4b5.jpg")
#  dataw = imageio.imread("w7v4b5.jpg")
#   datax = imageio.imread("w7v4b5.jpg")
#  datay = imageio.imread("w7v4b5.jpg")
#   dataz = imageio.imread("w7v4b5.jpg")
#  datafff = imageio.imread("w7v4b5.jpg")



   #datab = cv2.imread(Imagefile[z])
   #datac = cv2.imread(Imagefile[z])
   #datad = cv2.imread(Imagefile[z])
   #datae = cv2.imread(Imagefile[z])
   #dataf = cv2.imread(Imagefile[z])
   #datag = cv2.imread(Imagefile[z])
   #datah = cv2.imread(Imagefile[z])
   #datai = cv2.imread(Imagefile[z])
   #dataj = cv2.imread(Imagefile[z])
   #datak = cv2.imread(Imagefile[z])
   #datal = cv2.imread(Imagefile[z])
   #datam = cv2.imread(Imagefile[z])
   #datan = cv2.imread(Imagefile[z])
   #datao = cv2.imread(Imagefile[z])
   #datap = cv2.imread(Imagefile[z])
   #dataq = cv2.imread(Imagefile[z])
   #datar = cv2.imread(Imagefile[z])
   #datas = cv2.imread(Imagefile[z])
   #datat = cv2.imread(Imagefile[z])
   #datau = cv2.imread(Imagefile[z])
   #datav = cv2.imread(Imagefile[z])
   #dataw = cv2.imread(Imagefile[z])
   #datax = cv2.imread(Imagefile[z])
   #datay = cv2.imread(Imagefile[z])
   #dataz = cv2.imread(Imagefile[z])
   #datafff = cv2.imread(Imagefile[z])

   gray = cv2.cvtColor(databb, cv2.COLOR_BGR2GRAY)
   print(databb)
   print(databb.size)
   print(databb.shape)
   #Display Image
   #img = im.fromarray(datab, 'RGB')
   #img.show()
   #print(gray)
   
 
   imgtest = databb[700:1000,330:540]
   imgtest2 = im.fromarray(imgtest, 'RGB')
   imgtest2.show()
   print("Img Test: ")
   
   datalabel = np.load(Imagefile3[z],allow_pickle=True)
   #ORANGE THRESHOLDING - [255 126  18]
   #YELLOW THRESHOLDING - [255 195  26]
   #BROWN THRESHOLDING - [60 45  4]

   #print(imgtest[1,1,:])
   #[118 114 109] [59 44  5] [108 111 118]
   
#print(datab[620,640,:]) #[194 200 229]
#print(imgtest[1,100,:]) [42 248 254][82 83 74]
#DDEEW


#print(datab[120,790,:])
#print(datab[160,790,:])

         
#   for i in range(0,1):
#      data = np.load(Imagefile[z],allow_pickle=True)
#      img = im.fromarray(data, 'RGB')
#      print("Test Data")
#      print(data.size)
   [a,b] = datalabel.shape
#      print(data)




#   Overlay label on Image
 #  data2a = data
  # count12 = 0
   #for i in range(0,a):
    #  for j in range(0,b):
     #    if(data[i,j] == 2):
      #      count12 = count12 + 1
            #print(data[i,j])

  # print("count12: ")
  # print(count12)
  # print("DONE")




   #Overlay label on Image
   data2 = datab
   count = 0
   for i in range(0,a):
      for j in range(0,b):
         if(datalabel[i,j] != 0):
            data2[i,j] = 255
            datac[i,j,:] = 255;
            count = count + 1

   #Display Labelled Image
   print("Count: ")
   print(count)
 #  print(data2)
   img2 = im.fromarray(data2, 'RGB')
   img2.show()

   plt.imshow(data2, cmap='gray')
   plt.show()
