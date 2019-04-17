#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:26 2019

@author: MUT

Description create the the Training and Validation Dataset with directorys 
for the Model It creates from Images and Experimental eye movement 
data KDE masks which is used as a target for the model. 


"""

from modules.library import get_img_paths,get_dat_paths
import pandas as pd
from scipy import ndimage

from scipy import misc
import numpy as np
from tqdm import tqdm
import os
import re 
import seaborn as sns
import subprocess
import filecmp
import shutil

from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt




#bilderdaten
dir_images = os.getcwd()+'/images'
#blickdaten
dir_fixations = os.getcwd()+'/blickdaten'
#output
output_path = os.getcwd()+"/dataset"
if not os.path.exists(output_path):
  os.makedirs(output_path)
    
#monitor
my_dpi = 90
#path length for ploting kde images
pad_length = 27
#widh and height
#w = 1024+pad_length
#h = 768+pad_length

w = 1024
h = 768


#train validation split ratio
train_val_split = 0.1
#bandwith parameter for the KDE Masks
bw =30
#
#filtered_data = load_fixationData(dir_fixations)  
#image_objects =  get_image_objects(filtered_data,dir_images)  
#
#image = image_objects[0]
#
#sns.set_style("darkgrid", {"axes.facecolor": ".9",
#                         'figure.facecolor': 'white',
#                          'axes.spines.bottom': False,
#                           'axes.spines.left': False,
#                           'axes.spines.right': False,
#                           'axes.spines.top': False,
#
#                         })
#
#plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
#fig = sns.kdeplot(image[0].iloc[:,2],
#            image[0].iloc[:,3], 
#            kernel = "gau",
#            bw= 30,
#            cmap = plt.cm.gray,
#            shade=True, 
#            n_levels = 100,
#            legend= False,
#            cumulative= False)
#fig.axes.get_yaxis().set_visible(False)
#fig.axes.get_xaxis().set_visible(False)
#fig.axes.invert_yaxis()
#image_id = os.path.basename(image[-2])
#plt.tight_layout()
#plt.savefig(image_id,
#            dpi=my_dpi,
#            transparent='True',
#            bbox_inches='tight',
#            pad_inches = 0)
#
#
#plt.setp([fig.get_xticklines() + 
#          fig.get_yticklines() + 
#          fig.get_xgridlines() + 
#          fig.get_ygridlines()],antialiased=False)
#
#figure = fig.get_figure()
#
#figure.tight_layout(pad=0)
#figure.canvas.draw()
#
#
#image1 = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
#image1 = image1.reshape(figure.canvas.get_width_height()[::-1] + (3,))
#
#
#misc.imsave('outfile.png', image1[:,:,0])
##
##plt.imshow(image1)
#



def load_fixationData(dir_fixations):
  """load the dat, filter non valid and selecte important columns
  """
  fixation_paths = get_dat_paths(dir_fixations)
  all_data_list = [pd.read_table(fixation_path,encoding = "ISO-8859-1") 
                          for i,fixation_path in enumerate(fixation_paths) ]

  filtered_data = []
  for i in range(len(all_data_list)):
   filtered_data.append( all_data_list[i][(
        all_data_list[i]["fixinvalid"] != 1)  &
    (all_data_list[i]["sacinvalid"] != 1)  
    ].loc[:,["subject","colorimages","imageid","masktype","maskregion",
    "fixposx","fixposy","targetpresent", "expectedlocation", "fixno"]])
  
  return filtered_data

def seperate(filtered_data):
  """seperate different experiments and save it in a list
  
  choose the data associated to different experiment settings

    Arguments:
        data: DataFram that get filtered
        Experiment type Filter
        colorimages: 1 oder 0 (color oder grayscale images)
        masktype: 0, 1, oder 2 (control, low-pass oder high-pass filter)
        maskregion: 0, 1 oder 2 (control, periphery oder center) 
        
        Daraus ergibt sich entsprechend:
        masktype == 0 & maskregion == 0: Kontrollbedingung
        masktype == 1 & maskregion == 1: peripherer Tiefpassfilter
        masktype == 2 & maskregion == 1: peripherer Hochpassfilter
        masktype == 1 & maskregion == 2: zentraler Tiefpassfilter
        masktype == 2 & maskregion == 2: zentraler Hochpassfilter
  """
  list_of_exeperiments = []
  
  for i in range(len(filtered_data)):
    
    if i == 0: 
      ex = "memory" 
    
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages", "fixno",
                 "masktype","maskregion"
                 
                 ]],
      "original","color",
      ex ]
    
      )
      list_of_exeperiments.append([
      #pt,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "periphererTiefpass","color", ex ]
      )
    
      list_of_exeperiments.append([
      #ph,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "periphererHochpass","color" , ex ]
      )
      list_of_exeperiments.append([
      #zt,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "zentralerTiefpass","color" , ex ]
      )
      list_of_exeperiments.append([
      #zh,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "zentralerHochpass","color", ex ]
      )
      list_of_exeperiments.append([
      #controll,bw
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "original","grayscale", ex ]
      )
      list_of_exeperiments.append([
      #pt,grayscale
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "periphererTiefpass","grayscale", ex ]
      )
      list_of_exeperiments.append([
      #ph,grayscale
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "periphererHochpass","grayscale", ex ]
      )
      list_of_exeperiments.append([
      #zt,grayscale
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "zentralerTiefpass","grayscale", ex ])
      list_of_exeperiments.append([
      #zh,grayscale
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion"]],
      "zentralerHochpass","grayscale", ex ])
    elif i == 1: 
      ex = "search" 
      
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "original","color",ex,"target_not_present","non_expectedlocation" ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "original","color",ex,"targetpresent","non_expectedlocation" ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "original","color",ex,"targetpresent","expectedlocation" ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererTiefpass","color",ex,
      "target_not_present","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererTiefpass","color",ex,"targetpresent","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererTiefpass","color",ex,"targetpresent","expectedlocation" ])

      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererHochpass","color",ex,"target_not_present",
      "non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererHochpass","color",ex,"targetpresent","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererHochpass","color",ex,"targetpresent","expectedlocation" ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerTiefpass","color",ex,
      "target_not_present","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerTiefpass","color",ex,"targetpresent",
      "non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerTiefpass","color",ex,"targetpresent","expectedlocation" ])

      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerHochpass","color",ex,
      "target_not_present","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerHochpass","color",ex,
      "targetpresent","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 1) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerHochpass","color",ex,
      "targetpresent","expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "original","grayscale",ex,
      "target_not_present","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "original","grayscale",ex,"targetpresent","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 0) & 
                 (filtered_data[i]["maskregion"] == 0) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "original","grayscale",ex,"targetpresent","expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererTiefpass","grayscale",ex,
      "target_not_present","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererTiefpass","grayscale",ex,
      "targetpresent","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererTiefpass","grayscale",ex,
      "targetpresent","expectedlocation"])

      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererHochpass","grayscale",ex,
      "target_not_present","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererHochpass","grayscale",ex,
      "targetpresent","non_expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 1) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "periphererHochpass","grayscale",ex,"targetpresent","expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerTiefpass","grayscale",ex,"target_not_present",
      "non_expectedlocation" ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerTiefpass","grayscale",ex,
      "targetpresent","non_expectedlocation" ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 1) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerTiefpass","grayscale",ex,"targetpresent","expectedlocation"])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 0) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerHochpass","grayscale",ex,"target_not_present",
      "non_expectedlocation"
       ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 0)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerHochpass","grayscale", ex,
      "targetpresent","non_expectedlocation"
      ])
  
      list_of_exeperiments.append([
      #controll,color
      filtered_data[i][ (filtered_data[i]["colorimages"] == 0) &
                 (filtered_data[i]["masktype"] == 2) & 
                 (filtered_data[i]["maskregion"] == 2) &
                 (filtered_data[i]["targetpresent"] == 1) &
                 (filtered_data[i]["expectedlocation"] == 1)
                 ].loc[:,["subject","imageid","fixposx","fixposy",
                 "colorimages","fixno",
                 "masktype","maskregion",
                 "targetpresent", "expectedlocation"]],
      "zentralerHochpass","grayscale",ex,"targetpresent","expectedlocation"
       ]) 
  return list_of_exeperiments


def increment_filename(filename, marker="-"):
    """Appends a counter to a filename, or increments an existing counter."""
    basename, fileext = os.path.splitext(filename)

    # If there isn't a counter already, then append one
    if marker not in basename:
        components = [basename, 1, fileext]

    # If it looks like there might be a counter, then try to coerce it to an
    # integer and increment it. If that fails, then just append a new counter.
    else:
        base, counter = basename.rsplit(marker, 1)
        try:
            new_counter = int(counter) + 1
            components = [base, new_counter, fileext]
        except ValueError:
            components = [base, 1, fileext]

    # Drop in the marker before the counter
    components.insert(1, marker)

    new_filename = "%s%s%d%s" % tuple(components)
    return new_filename

def copyfile(src, dst):
    """Copies a file from path src to path dst.

    If a file already exists at dst, it will not be overwritten, but:

     * If it is the same as the source file, do nothing
     * If it is different to the source file, pick a new name for the copy that
       is distinct and unused, then copy the file there.

    Returns the path to the copy.
    """
    if not os.path.exists(src):
        raise ValueError("Source file does not exist: {}".format(src))

    # Create a folder for dst if one does not already exist
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    # Keep trying to copy the file until it works
    while True:

        # If there is no file of the same name at the destination path, copy
        # to the destination
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            return dst

        # If the namesake is the same as the source file, then we don't need to
        # do anything else
        if filecmp.cmp(src, dst):
            return dst

        # There is a namesake which is different to the source file, so pick a
        # new destination path
        dst = increment_filename(dst)

    return dst
  
def get_image_objects(filtered_data,dir_images):
  """creating objects each pictures with different experiment types
    
    
  input: filtered_data:      Dataframe of the Experients, 
         dir_images:         Directory of the images
         
         
  return: objects of each image with eyemovements and pictures

  """
  #get the image paths
  image_paths = get_img_paths(dir_images)
  
  original_paths = [i for i in image_paths if "original" in i and "._O" not in i]
  memory_paths = [i for i in original_paths if "memory" in i]
  search_paths = [i for i in original_paths if "search" in i]
  list_of_exp = seperate(filtered_data)

  list_of_masks = []
  
  for u in range(len(list_of_exp)):
  
    for i in range(len(np.unique(list_of_exp[u][0]["imageid"]) )):   
      if list_of_exp[u][3] == "memory":    
    
        for z, image_path in enumerate(memory_paths):

          if  list_of_exp[u][2] in image_path:
        
            if os.path.basename(image_path) ==  str(i+1)+".png":
              path = image_path
                
      elif list_of_exp[u][3] == "search":     
        if list_of_exp[u][4] == 'target_not_present':
          match = "L_"
        else:
          if list_of_exp[u][5] == 'expectedlocation':
            match = "E_"
          else:
            match = "U_"
        
        path_0 = [f for f in search_paths if (i+1) == int(re.findall(r'\d+', 
          os.path.basename(f))[0]) and match in f and list_of_exp[u][2] in f]
  
        try:
          path =path_0[0]
        except:
          pass        
      
      dataframe_temp = list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1]
      
      if dataframe_temp.shape[0]: 
        if list_of_exp[u][3] == "memory":
          list_of_masks.append(
                [list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1],
                list_of_exp[u][1],
                list_of_exp[u][2],
                list_of_exp[u][3],
                path,
                dataframe_temp.shape[0]])
        elif list_of_exp[u][3] == "search":
  
          list_of_masks.append(
                [list_of_exp[u][0][(list_of_exp[u][0]["imageid"] == i+1) ],
                list_of_exp[u][1],
                list_of_exp[u][2],
                list_of_exp[u][3],
                list_of_exp[u][4],
                list_of_exp[u][5],
                path,
                dataframe_temp.shape[0]]) 
  
  return list_of_masks


def create_images(each_images,
                  bw=30,
                  train_val_split = 0.1):
  
 
  
  """creating folder for different experiment type and also
  
  creting kernel density esitmaion masks for each picture
  
  parameter: 
   each_images     : objects
   bw              : bandwith
   train_val_split : splitting ratio
  """
  sns.set_style("darkgrid", {"axes.facecolor": ".9",
                           'figure.facecolor': 'white',
                            'axes.spines.bottom': False,
                             'axes.spines.left': False,
                             'axes.spines.right': False,
                             'axes.spines.top': False,
  
                           })

  counter = 1
  input_images_dir_name = "input_images" 
  label_images_dir_name = "masks" 
    
  splitter = round(1 / train_val_split) 
  
  for i, image in enumerate(each_images):
    
    
    image_id = os.path.basename(image[-2])
    
    image_id = image[1]+"_" +image[2]+"_" +image[3]+"_" +image_id

    os.chdir(output_path)
    if not os.path.isdir(image[2] + "_" +image[3]):  
      subprocess.call("mkdir " +image[2] + "_" +image[3], shell = True )
      
    #image_id = os.path.basename(image[-2]) 

    if not os.path.isdir(image[1]): 
      subprocess.call("mkdir " +image[2] + "_" +image[3]+"/"+image[1], shell = True )
    
    if (counter) % splitter == 0:
      directory_name = "_validation"
    else:
      directory_name = "_training"
      
    if not os.path.isdir(image[2] + "_" +image[3]+"/"+image[1]+"/"+input_images_dir_name+directory_name): 
      subprocess.call("mkdir "+image[2] + "_" +image[3]+"/"+image[1]+"/"+input_images_dir_name+ directory_name, 
                      shell = True )

    copyfile(image[-2], image[2] + "_" +image[3]+"/"+image[1]+"/"+input_images_dir_name+ directory_name+"/"+image_id) 
    
    if not os.path.isdir(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+ directory_name): 
      subprocess.call("mkdir "+image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+ directory_name , 
                      shell = True )
    
    counter = counter +1
    plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    fig = sns.kdeplot(image[0].iloc[:,2],
                image[0].iloc[:,3], 
                kernel = "gau",
                bw= 30,
                cmap = plt.cm.gray,
                shade=True, 
                n_levels = 100,
                legend= False,
                cumulative= False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.invert_yaxis()
    
    plt.tight_layout()
#    plt.savefig(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+directory_name +"/"+image_id,
#                dpi=my_dpi,
#                transparent='True',
#                bbox_inches='tight',
#                pad_inches = 0)
    
    plt.setp([fig.get_xticklines() + 
          fig.get_yticklines() + 
          fig.get_xgridlines() + 
          fig.get_ygridlines()],antialiased=False)

    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    


    image1 = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    image1 = image1.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    
    
    misc.imsave(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+directory_name +"/"+image_id, image1[:,:,0])
      
def get_image_objects(filtered_data,dir_images):
  """creating objects each pictures with different experiment types
    
    
  input: filtered_data:      Dataframe of the Experients, 
         dir_images:         Directory of the images
         
         
  return: objects of each image with eyemovements and pictures

  """
  #get the image paths
  image_paths = get_img_paths(dir_images)
  
  original_paths = [i for i in image_paths if "original" in i and "._O" not in i]
  memory_paths = [i for i in original_paths if "memory" in i]
  search_paths = [i for i in original_paths if "search" in i]
  list_of_exp = seperate(filtered_data)

  list_of_masks = []
  
  for u in range(len(list_of_exp)):
  
    for i in range(len(np.unique(list_of_exp[u][0]["imageid"]) )):   
      if list_of_exp[u][3] == "memory":    
    
        for z, image_path in enumerate(memory_paths):

          if  list_of_exp[u][2] in image_path:
        
            if os.path.basename(image_path) ==  str(i+1)+".png":
              path = image_path
                
      elif list_of_exp[u][3] == "search":     
        if list_of_exp[u][4] == 'target_not_present':
          match = "L_"
        else:
          if list_of_exp[u][5] == 'expectedlocation':
            match = "E_"
          else:
            match = "U_"
        
        path_0 = [f for f in search_paths if (i+1) == int(re.findall(r'\d+', 
          os.path.basename(f))[0]) and match in f and list_of_exp[u][2] in f]
  
        try:
          path =path_0[0]
        except:
          pass        
      
      dataframe_temp = list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1]
      
      if dataframe_temp.shape[0]: 
        if list_of_exp[u][3] == "memory":
          list_of_masks.append(
                [list_of_exp[u][0][list_of_exp[u][0]["imageid"] == i+1],
                list_of_exp[u][1],
                list_of_exp[u][2],
                list_of_exp[u][3],
                path,
                dataframe_temp.shape[0]])
        elif list_of_exp[u][3] == "search":
  
          list_of_masks.append(
                [list_of_exp[u][0][(list_of_exp[u][0]["imageid"] == i+1) ],
                list_of_exp[u][1],
                list_of_exp[u][2],
                list_of_exp[u][3],
                list_of_exp[u][4],
                list_of_exp[u][5],
                path,
                dataframe_temp.shape[0]]) 
  
  return list_of_masks


def create_images_in_one_folder(each_images,
                  bw=30,
                  train_val_split = 0.1):
  
 
  
  """creating folder for different experiment type and also
  
  creting kernel density esitmaion masks for each picture
  
  parameter: 
   each_images     : objects
   bw              : bandwith
   train_val_split : splitting ratio
  """
  sns.set_style("darkgrid", {"axes.facecolor": ".9",
                           'figure.facecolor': 'white',
                            'axes.spines.bottom': False,
                             'axes.spines.left': False,
                             'axes.spines.right': False,
                             'axes.spines.top': False,
  
                           })

  counter = 1

    
  splitter = round(1 / train_val_split) 
  
  for i, image in enumerate(each_images):
    
    
    image_id = os.path.basename(image[-2])
    
    image_id = image[1]+"_" +image[2]+"_" +image[3]+"_" +image_id

    os.chdir(output_path)
    
    if not os.path.isdir("training"):  
      subprocess.call("mkdir training", shell = True )
      
    if not os.path.isdir("validation"):  
      subprocess.call("mkdir validation", shell = True )
      
    if not os.path.isdir("training/images"):  
      subprocess.call("mkdir training/images", shell = True )  
      
    if not os.path.isdir("training/masks"):  
      subprocess.call("mkdir training/masks", shell = True )        

    if not os.path.isdir("validation/images"):  
      subprocess.call("mkdir validation/images", shell = True )  
      
    if not os.path.isdir("validation/masks"):  
      subprocess.call("mkdir validation/masks", shell = True )  
      

    if (counter) % splitter == 0:
      directory_name = "validation"
    else:
      directory_name = "training"
      

    copyfile(image[-2], directory_name+"/images/"+image_id) 
    
    
    counter = counter +1
    plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    fig = sns.kdeplot(image[0].iloc[:,2],
                image[0].iloc[:,3], 
                kernel = "gau",
                bw= 30,
                cmap = plt.cm.gray,
                shade=True, 
                n_levels = 100,
                legend= False,
                cumulative= False)
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.invert_yaxis()
    
    plt.tight_layout()
#    plt.savefig(image[2] + "_" +image[3]+"/"+image[1]+"/"+label_images_dir_name+directory_name +"/"+image_id,
#                dpi=my_dpi,
#                transparent='True',
#                bbox_inches='tight',
#                pad_inches = 0)
    
    plt.setp([fig.get_xticklines() + 
          fig.get_yticklines() + 
          fig.get_xgridlines() + 
          fig.get_ygridlines()],antialiased=False)

    figure = fig.get_figure()
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    


    image1 = np.fromstring(figure.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    image1 = image1.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    
    
    misc.imsave(directory_name+"/masks/"+image_id, image1[:,:,0])
    
    
    
def rotate_allimages(path, degree):
  """Rotate every images in the given directory, with the given degree
  
  input: directory path, degree to rotate
  """
  image_paths = get_img_paths(path)  
  
  # iterate through the names of contents of the folder
  for image_path in image_paths:
    # create the full input path and read the file
    image_to_rotate = ndimage.imread(image_path)
    # rotate the image
    rotated = ndimage.rotate(image_to_rotate, degree) 
    # create full output path, 'example.jpg' 
    # becomes 'rotate_example.jpg', save the file to disk
    #fullpath = os.path.join(outPath, 'rotated_'+image_id)
    misc.imsave(image_path, rotated)
    
    
filtered_data = load_fixationData(dir_fixations)
  
#save each pictures with experiment as objects 
image_objects =  get_image_objects(filtered_data,dir_images)



#create_images(image_objects, bw, train_val_split)


create_images_in_one_folder(image_objects, bw, train_val_split)
 
#def main():
# 
#  #load the fixationdata
#  filtered_data = load_fixationData(dir_fixations)
#    
#  #save each pictures with experiment as objects 
#  image_objects =  get_image_objects(filtered_data,dir_images)
#      
#  #create kde images with bandwith and for each class own directory
#  create_images(image_objects, bw, train_val_split)
#  
#  #rotate all images in the  give path and degree
#  #rotate_allimages(output_path, 90)
#    
#if __name__ == '__main__':
#  main()
  
    