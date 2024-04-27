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

            
#Load 362 Images
Imagefile = ["de6b2u.jpg", "gc8f2h.jpg", "g8uwmh.jpg", "fmdhxu.jpg", "epy8zd.jpg", "e5opgj.jpg", "dwl1q6.jpg", "de6b2u.jpg", "ct4d2i.jpg", "cg697z.jpg", "795qky.jpg", "453g1w.jpg", "79h0ma.jpg", "52p9ah.jpg", "52p9ah.jpg", "45zs2b.jpg","24eh8r.jpg", "20di1f.jpg", "17gw5j.jpg", "7iatmd.jpg", "6ge1y0.jpg", "3gpbh5.jpg", "zu6ojf.jpg", "1knjzt.jpg", "zu6ojf.jpg", "zsr74n.jpg", "zye2k6.jpg", "zu6ojf.jpg"  
             ]
         
Imagefile2 = ["de6b2u_gt.npy", "gc8f2h_gt.npy", "g8uwmh_gt.npy", "fmdhxu_gt.npy", "epy8zd_gt.npy", "e5opgj_gt.npy", "dwl1q6_gt.npy", "de6b2u_gt.npy", "ct4d2i_gt.npy", "cg697z_gt.npy", "795qky_gt.npy", "453g1_gt.npy", "79h0ma_gt.npy", "52p9ah_gt.npy", "45zs2b_gt.npy", "24eh8r_gt.npy", "20di1f_gt.npy", "17gw5j_gt.npy","7iatmd_gt.npy", "6ge1y0_gt.npy", "3gpbh5_gt.npy", "zu6ojf_gt.npy", "1knjzt_gt.jpg","zu6ojf_gt.npy", "zye2k6_gt.npy",  "zu6ojf_gt.npy"
             ]
         
for i in range(0,1):
   #Read in Image
   datab = cv2.imread(Imagefile[i])
   datac = cv2.imread(Imagefile[i])
   datad = cv2.imread(Imagefile[i])
   datae = cv2.imread(Imagefile[i])
   dataf = cv2.imread(Imagefile[i])
   datag = cv2.imread(Imagefile[i])
   datah = cv2.imread(Imagefile[i])
   datai = cv2.imread(Imagefile[i])
   dataj = cv2.imread(Imagefile[i])
   datak = cv2.imread(Imagefile[i])
   datal = cv2.imread(Imagefile[i])
   datam = cv2.imread(Imagefile[i])
   datan = cv2.imread(Imagefile[i])
   datao = cv2.imread(Imagefile[i])
   datap = cv2.imread(Imagefile[i])
   dataq = cv2.imread(Imagefile[i])
   datar = cv2.imread(Imagefile[i])
   datas = cv2.imread(Imagefile[i])
   datat = cv2.imread(Imagefile[i])
   datau = cv2.imread(Imagefile[i])
   datav = cv2.imread(Imagefile[i])
   dataw = cv2.imread(Imagefile[i])
   datax = cv2.imread(Imagefile[i])
   datay = cv2.imread(Imagefile[i])
   dataz = cv2.imread(Imagefile[i])
   datafff = cv2.imread(Imagefile[i])
   gray = cv2.cvtColor(datab, cv2.COLOR_BGR2GRAY)
   print(datab)
   print(datab.size)
   print(datab.shape)
   #Display Image
   img = im.fromarray(datab, 'RGB')
   img.show()
   #print(gray)

#imgtest = datab[120:220,680:790]
#imgtest2 = im.fromarray(imgtest, 'RGB')
#imgtest2.show()
print(datab[120,790,:])
fdfd
         
for i in range(0,1):
   data = np.load('de6b2u_gt.npy',allow_pickle=True)
   #img = im.fromarray(data, 'RGB')
   print(data.size)
   [a,b] = data.shape
   print(data)
   np.save('start.txt',data)

#Overlay label on Image
data2 = data
count = 0
for i in range(0,a):
   for j in range(0,b):
      if(data[i,j] != 0):
         data2[i,j] = 255
         datac[i,j,:] = 255;
         count = count + 1

#Display Labelled Image
print("Count: ")
print(count)
print(data2)
img2 = im.fromarray(datac, 'RGB')
img2.show()
 

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
img3.show()

img4 = im.fromarray(data3, 'RGB')
img4.show()

img5 = im.fromarray(data5, 'RGB')
img5.show()


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
img7.show()



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





one = 0
two = 0

for j in range(0,a1):
   if(y.values[j,0] > 1000):
      gray1[linecounttest,0] = y.values[j,0]
      linecounttest = linecounttest + 1
      gray1b[linecounttest] = j
      #one = y.values[j,1]),1][j,0]
      #two = y.values[int(gray1[0,0]),1][j,1]
      #gray2[linecounttest,0] = one
      #gray2[linecounttest,1] = two

 
print("linecounttest")
print(linecounttest)
#print("one")
#print(gray2[:,0])
#print("two:")
#print(gray2[:,1])



img8 = im.fromarray(np.uint8(graytest3 * 255), 'L')
img8.show()
for m in range(0,linecounttest):
   [size1, size2] = y.values[int(gray1b[m]),1].shape
   print("starting")
   print(size1)
   print(size2)
   for p in range(0, size1-1):
   #img8.putpixel((int(y.values[int(gray1b[j]),1][0,0]),int(y.values[int(gray1b[j]),1][0,1])), 200)
      img8.putpixel((int(y.values[int(gray1b[m]),1][p,0]),int(y.values[int(gray1b[m]),1][p,1])), 255)

      #print("starting 2")

#print(y.values[29,1][0,0])
print("final image:")
img8.show()



#Transposing to right size. Transpose and then Rotate to see Final Image
img8.show()
img9 = img8.transpose(1)
img9.show()
img10 = img9.rotate(-90)
img10.show()



print("Liptinite Dark Gray Processing")
#Color Thresholding for Liptinite - Dark Gray,
# Ideal Values: [149 151 145]
[a1,b1,c1] = datab.shape
print(a1)
print(b1)
print(c1)
for i in range(0,a1):
   for j in range(0,b1):
      if((datae[i,j,0] > 148) & (datae[i,j,0] < 150)):
         if((datae[i,j,1] > 150) & (datae[i,j,1] < 152)):
            if( ((datae[i,j,2] > 144) & (datae[i,j,2] <146))):# | ((datab[i,j,2] > 40) & (datab[i,j,2] < 60)) ):
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


print("Final Liptrinite Dark Gray Image")           
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
   print(y.values[j,0])
   if(y.values[j,0] > 1000):
      gray1[linecounttest,0] = y.values[j,0]
      linecounttest = linecounttest + 1
      gray1b[linecounttest] = j
      #one = y.values[j,1]),1][j,0]
      #two = y.values[int(gray1[0,0]),1][j,1]
      #gray2[linecounttest,0] = one
      #gray2[linecounttest,1] = two

 
print("linecounttest for 2")
print(linecounttest)
#print("one")
#print(gray2[:,0])
#print("two:")
#print(gray2[:,1])



img8 = im.fromarray(np.uint8(graytest3 * 255), 'L')
img8.show()
for m in range(0,linecounttest):
   [size1, size2] = y.values[int(gray1b[m]),1].shape
   print("starting")
   print(size1)
   print(size2)
   for p in range(0, size1-1):
   #img8.putpixel((int(y.values[int(gray1b[j]),1][0,0]),int(y.values[int(gray1b[j]),1][0,1])), 200)
      img8.putpixel((int(y.values[int(gray1b[m]),1][p,0]),int(y.values[int(gray1b[m]),1][p,1])), 255)

      #print("starting 2")

#print(y.values[29,1][0,0])
print("final image 2 for Liptinite:")
img8.show()



#Transposing to right size. Transpose and then Rotate to see Final Image
img8.show()
img9 = img8.transpose(1)
img9.show()
img10 = img9.rotate(-90)
img10.show()



#Do the same thing for the other macerels:
# (Vitrinite - Medium to Light Gray) & (Inertinite - White)
#Next: Vitrinite - Medium to Light Gray - April 26, 2024


#Color Thresholding for Vitrinite - Medium to Light Gray

sss
xco = y.values[int(gray1[0,0]),1][i,0]

yco = y.values[int(gray1[0,0]),1][i,1]
threshold4[xco,yco] = 5


#complete image
for i in range(0,d-1):
    for j in range(0,e-1):
        finaldata[i,j,l] =  int(graytest2.getpixel((i,j)))

 
img8.show()

#      gray2[j,0] = y.values[int(gray1[0,0]),1][i,0]
 #     gray2[j,1] = y.values[int(gray1[0,0]),1][i,1]
  #    threshold[xco,yco] = 1

print("linecounttest: ")
print(linecounttest)
print("gray2: ")
print(gray2)

   
#Store LineCounts in Total Line Array
TotalRegions = linecounttest

regions2 = regionprops(graytest)
imshow(regions2)
regions2.show()


        
#final image
for i in range(0,d-1):
   for j in range(0,e-1):
      graytest2.putpixel((i,j), 0)
      if(threshold[j,i] == 1):
         graytest2.putpixel((i,j), 255)


graytest2.show()


#Next: Area thresholding to > 10 um
# Get Final Image
# Add Label on top





#secondone = dataff[80:340, 800:920]
#img6 = im.fromarray(secondone, 'RGB')
#img6.show()
ddd
#Variable to hold image size
xycoordinates = np.zeros((362,2))

#Array Variable to hold Average Intensity of all Image Corpus Images
AverageIntensity = np.zeros((362,1))

#Array Variable to hold Color Homogeneity Percentage
HomogeneityPercent = np.zeros((362,1))


#Array Variable to hold Number of Lines Detected
TotalLines = np.zeros((362,1))

#Array Variable to hold Mean PSD of Images
PSDMean = np.zeros((362,1))

#Array Variable to hold Entropy Mean of Images
EntropyMean = np.zeros((362,1))

#Array Variable to hold All One Shot Algorithm Development
OneShotAlgorithm = np.zeros((362,5))

#Array to Hold String Variable for file
StringFile = "{"

#One Shot Algorithm Development
#Average Intensity Value across the Image
#Color Homogeneity (for grayscale)
#Edge Filtration  of line detection(such as Canny Edge)
#Power Spectrum Density Cross Correlation across the Image
#Entropy across the Image


#for l in range(0,362):
    #Read in Image Corpus Image
for i in range(0,362):
   datab = cv2.imread(Imagefile[i])

   #Convert Image from RGB to Gray
   gray = cv2.cvtColor(datab, cv2.COLOR_BGR2GRAY)
   
   # Get the shape of the image [Row x Column] and save values
   #as row and colum variables
   [xycoordinates[i,0], xycoordinates[i,1]]= gray.shape
   row = int(xycoordinates[i,0])
   column = int(xycoordinates[i,1])

   #PROCESS 1: AVERAGE INTENSITY
   #Variable to hold Sum of Intensity across an Entire Image
   SumIntensity = 0
   
   #Nested Loop to Extract Average Intensity from the Image
   for j in range(0,row):
      for k in range(0,column):
         #Obtain Pixel Intensity
         test = gray[j,k]
         
         #Add Pixel Value to Sum of Intensity)
         SumIntensity = SumIntensity + test

   #Get the Average Intensity Value of the Grayscale Image
   AvIntensity = ((SumIntensity/row)/column)
   
   #Store the Average Intensity Value of the Grayscale Image in
   #the Average Intensity Array
   AverageIntensity[i] = AvIntensity 

       
   #PROCESS 2: COLOR HOMOGENEITY

   #Variable to Count Homogeneous Pixels
   HomogeneousCount = 0
   #Nested Loop to Extract Homogeneous Pixels
   for j in range(1,row-1):
      for k in range(1,column-1):
         #Obtain Homogeneous Value
         hv = (gray[j,k]+gray[j-1,k]+gray[j+1,k]+gray[j,k+1]+gray[j-1,k+1]+gray[j+1,k+1]+gray[j,k-1]+gray[j-1,k-1]+gray[j+1,k-1])
         hv2 = hv/9
         if((abs(gray[j,k] - hv2)) < 30):
            HomogeneousCount = HomogeneousCount + 1
   #Store Homogeneity Percent in Homogeneity Array
   HomogeneityPercent[i] = ((HomogeneousCount * 100)/(row * column))
         


   #PROCESS 3: LINE DETECTION

   #Variable to Count Line Detection
   line = 0
   lineimage = gray

   #Grayscale Thresholding to extract lines in Image
   for j in range(1,row-1):
       for k in range(1,column-1):
           #if( ( gray[j,k] > 50) &  (gray[j,k] < 80)):
           if( ( gray[j,k] > 180)):
              lineimage[j,k] = 255
           else:
              lineimage[j,k] = 0   

   #perform Region Props on Thresholded Image
   lineimagecc = np.array(lineimage)
   #Select Pixels Greater than 100 with a mask
   mask = lineimagecc > 100
   labels = measure.label(mask)

   #Segment out Regions
   regions = measure.regionprops(labels, lineimagecc)
   numlabels = len(regions)
   regions = regionprops_table(labels, properties=('area', 'coords'))
   #print(regions)
   pd.DataFrame(regions)
   y = pd.DataFrame(regions)
   #Get Shape and Size of Regions
   [a1,b1] = y.shape

   #Select Only Regions Greater than 500 Pixels and Get their Line Count
   linecount = 0

   for j in range(0,a1):
       if(y.values[j,0] > 200):
          linecount = linecount + 1


   #Store LineCounts in Total Line Array
   TotalLines[i] = linecount



   #PROCESS 4 POWER SPECTRUM DENSITY (PSD)

   #Variable to Hold PSD
   psd = 0
   psdimage = gray
   psdsum = 0

   #Get PSD
   fourier_image = np.fft.fftn(gray)
   fs = 1000.0 #1 kHz sampling frequency
   #signal = grayscale image
   signal = gray
   (S,f) = plt.psd(signal, Fs=fs)
   #f contains the frequency components
   #S is the PSD
   #plt.semilogy(f,S)
   #plt.xlim([0,100])
   #plt.xlabel('frequency [Hz]')
   #plt.ylabel('PSD [V**22222Hz]')
   #plt.show()

   #Size of the PSD
   psd =S.size

   #Get Average PSD
   for j in range(0,psd):
      psdsum = psdsum + S[j]

   #Store Average PSD in PSDMean Variable
   PSDMean[i] = psdsum/psd


   
   #PROCESS 5 ENTROPY

   #Variable to Hold Entropy
   entropyimage = gray
   entropysum = 0
   entropy1 = np.array(entropyimage)

   #Get Entropy Value
   entropy2 = entropy(entropy1, base=10)

   #Get Size of Entropy
   entropysize =entropy2.size

   #Find Entropy Mean
   for j in range(0,entropysize):
      entropysum = entropysum + entropy2[j]

   #Store Average Entropy in EntropyMean Variable
   EntropyMean[i] = entropysum/entropysize

   
print("Homogeneity% : ")
print(HomogeneityPercent)  
print("Array Final Average Intensity: ")
print(AverageIntensity)
print("Total Line Count: ")
print(TotalLines)
print("PSD Mean: ")
print(PSDMean)
print("Entropy Mean: ")
print(EntropyMean)


#insert the 5 Oneshot Algorithm Features into the OneShotAlgorithm Array Variable
OneShotAlgorithm[:,0] = HomogeneityPercent[:,0]
OneShotAlgorithm[:,1] = AverageIntensity[:,0]
OneShotAlgorithm[:,2] = TotalLines[:,0]
OneShotAlgorithm[:,3] = PSDMean[:,0]
OneShotAlgorithm[:,4] = EntropyMean[:,0]

#Print the OneshotAlgorithm Array Variable
print("OneshotAlgorithm : ")
print(OneShotAlgorithm)

#Access Oneshot Algorithm Array Variable Shape and Size
OneShotAlgorithm.shape
OneShotAlgorithm.size
        
print("XY Coordinates: ")     
print(xycoordinates)


#Upload Images to be used as Test Data
#Load 50 Images
ImageTestData = ["aogst.png", "bbqxg.png", "cwrzg.png", "dbxmq.png",
             "eckdo.png", "engqt.png", "ezuen.png", "fhnts.png",
             "gdchp.png", "hqfll.png", "iiqot.png", "ijdzo.png",
             "itzis.png", "iwiev.png" ,"iyphf.png", "jggsc.png",
             "jjvxo.png", "klxxh.png", "kthks.png", "lvoiu.png",
             "lzwdh.png", "mfros.png", "miufj.png", "mzwjh.png",
             "nfnmb.png", "ngxvb.png", "nojtp.png", "pijkw.png",
             "qemqk.png", "qsiio.png", "qtudi.png", "siisg.png",
             "sjplt.png", "skjpp.png", "skqhg.png", "uciie.png",
             "ukwfg.png", "uyjad.png", "vcnst.png", "wakcc.png",
             "wtdvm.png", "wvbsi.png", "xvhbx.png", "yjglq.png",
             "yracw.png", "yzaxb.png", "zivsv.png", "zjxrd.png",
             "zluym.png", "zqqan.png",

             ]


#Variable to hold image size of all Test Images
xycoordinatestest = np.zeros((50,2))

#Array Variable to hold Average Intensity of all Test Images
AverageIntensitytest = np.zeros((50,1))

#Array Variable to hold Color Homogeneity Percentages of all Test Images
HomogeneityPercenttest = np.zeros((50,1))


#Array Variable to hold Number of Lines Detected in all Test Images
TotalLinestest = np.zeros((50,1))

#Array Variable to hold Mean PSD of all Test Images
PSDMeantest = np.zeros((50,1))

#Array Variable to hold Entropy Mean of all Test Images
EntropyMeantest = np.zeros((50,1))

#Array Variable to hold All One Shot Algorithm Development of all Test Images
OneShotAlgorithmtest = np.zeros((50,5))

#for l in range(0,50):
    #Read in Test Images
for i in range(0,50):
   datac = cv2.imread(ImageTestData[i])

   #Convert Image from RGB to Gray
   graytest = cv2.cvtColor(datac, cv2.COLOR_BGR2GRAY)
   
   # Get the shape of the image [Row x Column] and save values
   #as row and colum variables
   [xycoordinatestest[i,0], xycoordinatestest[i,1]]= graytest.shape
   rowtest = int(xycoordinatestest[i,0])
   columntest = int(xycoordinatestest[i,1])

   #PROCESS 1: AVERAGE INTENSITY
   #Variable to hold Sum of Intensity across an Entire Test Image
   SumIntensitytest = 0
   
   #Nested Loop to Extract Average Intensity from the Test Image
   for j in range(0,rowtest):
      for k in range(0,columntest):
         #Obtain Pixel Intensity
         testtest = graytest[j,k]
         
         #Add Pixel Value to Sum of Intensity)
         SumIntensitytest = SumIntensitytest + testtest

   #Get the Average Intensity Value of the Grayscale Test Image
   AvIntensitytest = ((SumIntensitytest/rowtest)/columntest)
   
   #Store the Average Intensity Value of the Grayscale Test Image in
   #the Average Intensity Array
   AverageIntensitytest[i] = AvIntensitytest 

       
   #PROCESS 2: COLOR HOMOGENEITY

   #Variable to Count Homogeneous Pixels
   HomogeneousCounttest = 0
   #Nested Loop to Extract Homogeneous Pixels
   for j in range(1,rowtest-1):
      for k in range(1,columntest-1):
         #Obtain Homogeneous Value
         hvtest = (graytest[j,k]+graytest[j-1,k]+graytest[j+1,k]+graytest[j,k+1]+graytest[j-1,k+1]+graytest[j+1,k+1]+graytest[j,k-1]+graytest[j-1,k-1]+graytest[j+1,k-1])
         hv2test = hvtest/9
         if((abs(graytest[j,k] - hv2test)) < 30):
            HomogeneousCounttest = HomogeneousCounttest + 1
   #Store Homogeneity Percent in Homogeneity Array
   HomogeneityPercenttest[i] = ((HomogeneousCounttest * 100)/(rowtest * columntest))
         


   #PROCESS 3: LINE DETECTION

   #Variable to Count Line Detection
   linetest = 0
   lineimagetest = graytest

   #Grayscale Thresholding to extract lines in Test Image
   for j in range(1,rowtest-1):
       for k in range(1,columntest-1):
           #if( ( gray[j,k] > 50) &  (gray[j,k] < 80)):
           if( ( graytest[j,k] > 180)):
              lineimagetest[j,k] = 255
           else:
              lineimagetest[j,k] = 0   

   #perform Region Props on Thresholded Test Image
   lineimagecctest = np.array(lineimagetest)
   #Select Pixels Greater than 100 with a mask
   masktest = lineimagecctest > 100
   labelstest = measure.label(masktest)

   #Segment out Regions
   regionstest = measure.regionprops(labelstest, lineimagecctest)
   numlabelstest = len(regionstest)
   regionstest = regionprops_table(labelstest, properties=('area', 'coords'))
   #print(regions)
   pd.DataFrame(regionstest)
   y = pd.DataFrame(regionstest)
   #Get Shape and Size of Regions
   [a1,b1] = y.shape

   #Select Only Regions Greater than 500 Pixels and Get their Line Count
   linecounttest = 0

   for j in range(0,a1):
       if(y.values[j,0] > 200):
          linecounttest = linecounttest + 1


   #Store LineCounts in Total Line Array
   TotalLinestest[i] = linecounttest



   #PROCESS 4 POWER SPECTRUM DENSITY (PSD)

   #Variable to Hold PSD
   psdtest = 0
   psdimagetest = graytest
   psdsumtest = 0

   #Get PSD
   fourier_imagetest = np.fft.fftn(graytest)
   fstest = 1000.0 #1 kHz sampling frequency
   #signaltest = grayscale image
   signaltest = graytest
   (S,f) = plt.psd(signaltest, Fs=fstest)
   #f contains the frequency components
   #S is the PSD
   #plt.semilogy(f,S)
   #plt.xlim([0,100])
   #plt.xlabel('frequency [Hz]')
   #plt.ylabel('PSD [V**22222Hz]')
   #plt.show()

   #Size of the PSD
   psdtest =S.size

   #Get Average PSD
   for j in range(0,psdtest):
      psdsumtest = psdsumtest + S[j]

   #Store Average PSD in PSDMean Variable
   PSDMeantest[i] = psdsumtest/psdtest


   
   #PROCESS 5 ENTROPY

   #Variable to Hold Entropy
   entropyimagetest = graytest
   entropysumtest = 0
   entropy1test = np.array(entropyimagetest)

   #Get Entropy Value
   entropy2test = entropy(entropy1test, base=10)

   #Get Size of Entropy
   entropysizetest =entropy2test.size

   #Find Entropy Mean
   for j in range(0,entropysizetest):
      entropysumtest = entropysumtest + entropy2test[j]

   #Store Average Entropy in EntropyMean Variable
   EntropyMeantest[i] = entropysumtest/entropysizetest

   
   print("Homogeneity% for Test Image : ")
   print(HomogeneityPercenttest)  
   print("Array Final Average Intensity for Test Image: ")
   print(AverageIntensitytest)
   print("Total Line Count for Test Image: ")
   print(TotalLinestest)
   print("PSD Mean for Test Image: ")
   print(PSDMeantest)
   print("Entropy Mean for Test Image: ")
   print(EntropyMeantest)

   #insert the 5 Oneshot Algorithm Features into the OneShotAlgorithm Array Variable
   OneShotAlgorithmtest[:,0] = HomogeneityPercenttest[:,0]
   OneShotAlgorithmtest[:,1] = AverageIntensitytest[:,0]
   OneShotAlgorithmtest[:,2] = TotalLinestest[:,0]
   OneShotAlgorithmtest[:,3] = PSDMeantest[:,0]
   OneShotAlgorithmtest[:,4] = EntropyMeantest[:,0]

   #Print the OneshotAlgorithm Array Variable
   print("OneshotAlgorithm for Test Image: ")
   print(OneShotAlgorithmtest)

   #Access Oneshot Algorithm Array Variable Shape and Size
   OneShotAlgorithmtest.shape
   OneShotAlgorithmtest.size
           
   print("XY Coordinates for Test Image: ")     
   print(xycoordinatestest)
   print("Done for now")

   #Match Test Image against the 362 Images in Corpus Image and Extract the
   #Top Three (3) Best Fits

   #Array Variable to hold Matching Data per Test Image
   Fit = np.zeros((362,5))
   for j in range(0,362):
      Fit[j,0] = abs(OneShotAlgorithm[j,0] - OneShotAlgorithmtest[i,0])
      Fit[j,1] = abs(OneShotAlgorithm[j,1] - OneShotAlgorithmtest[i,1])
      Fit[j,2] = abs(OneShotAlgorithm[j,2] - OneShotAlgorithmtest[i,2])
      Fit[j,3] = abs(OneShotAlgorithm[j,3] - OneShotAlgorithmtest[i,3])
      Fit[j,4] = abs(OneShotAlgorithm[j,4] - OneShotAlgorithmtest[i,4])
        
   print("Fit to Oneshot Algorithm: ")     
   print(Fit)
   print("Done for now") 

     
   #Sum Fit to Oneshot Algorithm to find the Lowest three rows (best 3 fits)
   SumFit = np.zeros((362,1))
   TestSumFit = np.zeros((362,1))
   for j in range(0,362):
      SumFit[j,0] = sum(Fit[j,:])
      TestSumFit[j,0] = sum(Fit[j,:])    
   print("SumFit for Oneshot Algorithm: ")     
   print(SumFit)
   print("Done for now")
   
   MaxSumFit = max(TestSumFit)
   print("MaxSumFit")
   print(MaxSumFit)
   
   #Check for Lowest Three sum rows and ir confidences
   RowIndexSum = np.zeros((3,1))
   Match1 = min(TestSumFit)
   MatchConfidence = np.zeros((3,1))
   FitRow = 0
   for j in range(0,362):
      if(TestSumFit[j] == Match1):
         RowIndexSum[0] = j
         FitRow = j
   print("FitRow 1: ")
   print(FitRow)
   TestSumFit[FitRow] = 1000
   
   Match2 = min(TestSumFit)
   for j in range(0,362):
      if(TestSumFit[j] == Match2):
         RowIndexSum[1] = j
         FitRow = j
   TestSumFit[FitRow] = 1000
   print("FitRow 2: ")
   print(FitRow)
   
   Match3 = min(TestSumFit)
   for j in range(0,362):
      if(TestSumFit[j] == Match3):
         RowIndexSum[2] = j
         FitRow = j
   print("FitRow 3: ")
   print(FitRow)
   TestSumFit[FitRow] = 1000


   print("Three Rows: ")     
   print(RowIndexSum)
   print("The Three Rows: ")     
   print(SumFit[int(RowIndexSum[0])])
   print(SumFit[int(RowIndexSum[1])])
   print(SumFit[int(RowIndexSum[2])])

   MatchConfidence[0] = 1.0
   MatchConfidence[1] = (1 - ((SumFit[int(RowIndexSum[1])]) / MaxSumFit))
   MatchConfidence[2] = (1 - ((SumFit[int(RowIndexSum[2])]) / MaxSumFit))

   print("Match Confidences: ")     
    
   print(MatchConfidence[0])
   print(MatchConfidence[1])
   print(MatchConfidence[2])


   #Get confidence level for three images using the SumFit[int(RowIndexSum[2])])
   #data above tomorrow
   
   
   print("Selected Three Images: ")     
   print(Imagefile[int(RowIndexSum[0])])
   print(Imagefile[int(RowIndexSum[1])])
   print(Imagefile[int(RowIndexSum[2])])

   StringFile = StringFile + "'" + ImageTestData[i] + "': [{'label': '" + str(Imagefile[int(RowIndexSum[0])]) + "', 'confidence': 1.0}, {'label': '" + str(Imagefile[int(RowIndexSum[1])]) + "', 'confidence':" + str(MatchConfidence[1]) + "}, {'label': '" + str(Imagefile[int(RowIndexSum[2])]) + "', 'confidence':" + str(MatchConfidence[2]) + "}], "
   
   print("Addded String File")
   print(StringFile)

   conf1 = str(MatchConfidence[0])
   conf2 = str(MatchConfidence[1])
   conf3 = str(MatchConfidence[2])

   
   print("String Confidences :")
   print(conf1)
   print(conf2)
   print(conf3)
   #Store Selection in an Array String Variable to be concatenated for the
   # 50 Test Images
   
   print("Done for Matching Images Singular")     

StringFile = StringFile + "}"
print("Final String: ")
print(StringFile)
np.save('Onward3start.txt',StringFile)




#img10b = img8
#img10b = img10b.transpose(img8.FLIP_TOP_BOTTOM)
#img11b = img10b.transpose(img8.TRANSPOSE)
#img10b = img10b.transpose(img8.FLIP_LEFT_RIGHT)
#img11b = img10b.transpose(img8.TRANSVERSE)
#img10b = img10b.trans.TRANSVERSE


#img8bb = np.uint8(img8)
#img10a = img8bb.transpose(img8bb.FLIP_TOP_BOTTOM)
#img10aa = im.fromarray(np.uint8(image10a * 255), 'L')
#img10aa.show()

#img10aa = im.fromarray(np.uint8(flipimage1 * 255), 'L')

	
#img10aa = im.fromarray(np.uint8(image10a * 255), 'L')
#img10aa.show()

#img10aa = im.fromarray(np.uint8(flipimage1 * 255), 'L')
#img10aa.show()

#transposed_image = img8.transpose(-90)
#transposed_image.show()
#flipimage1 = cv2.flip(img8bb,0)
#flipimage1.show()
#flipimage2 = cv2.flip(img8,1)
#flipimage2.show()
#img11= img8b.transpose(TRANSPOSE)
#img11.show()
#img12= img8.transpose(TRANSVERSE)
#img12.show()


#img9 = img8.transpose(90)
#img9 = img8.rotate(90)
#img9.show()
#img8.transpose('flip')
#print(int(y.values[10,1][0,0])
#print(y.values[:,1])
