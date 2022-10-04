PASCAL_CLASSES = [
    'none',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

# Fill in the classes you want to retain
path = "/home/abdelrahman/Team/FairMOT/data/VOCdevkit/VOC2012/ImageSets/Main/train.txt"

my_file = open(path, "r")
  
# reading the file
data = my_file.read()
  
# replacing end of line('/n') with ' ' and
# splitting the text it further when '.' is seen.
data_into_list = data.split("\n")
  
# printing the data
my_file.close()
classesINeed = ['none', 'person']

# Define the relevant directories 

xmlDirectory = '/home/abdelrahman/Team/FairMOT/data/VOCdevkit/VOC2012/Annotations/'

modifiedXmlDir = '/home/abdelrahman/Team/FairMOT/data/VOCdevkit/VOC2012/newAnnotations/'

JPEGdirectory = '/home/abdelrahman/Team/FairMOT/data/VOCdevkit/VOC2012/JPEGImages/'

modifiedJPEGdir = '/home/abdelrahman/Team/FairMOT/data/VOCdevkit/VOC2012/newJPEGImages/'

listFile = '/home/abdelrahman/Team/FairMOT/data/VOCdevkit/VOC2012/trainval.txt'

labelMap = '/home/abdelrahman/Team/FairMOT/data/VOCdevkit/VOC2012/labelmap_voc.prototxt'

listfile = open(listFile, 'w')
labelmap = open(labelMap, 'w')

import os
from shutil import copyfile
from os.path import isfile, join

# Get all the xml files into list
onlyfiles = [f for f in os.listdir(xmlDirectory) if isfile(join(xmlDirectory,f))]

# For saving the class - file dictionary 
fileDict = {}

i = 0 

# for limiting number of images
imgnum = 0


for claz in classesINeed:
	fileDict[claz] = []
	# generate labelmap file
	labelmap.write('item {\n  name: "' + claz + '"\n  label: ' + str(i) + '\n  display_name: "' + claz + '"\n}\n')
	i += 1

labelmap.close()
# Parse each XML file
import xml.etree.ElementTree as ET

for filename in onlyfiles:
	filelink = join(xmlDirectory,filename)
	tree = ET.parse (filelink)
	root = tree.getroot()
	objs = root.findall('object')
	objNum  = 0
	for obj in objs:
		objNum += 1
		currentObj = obj.find('name').text
		if currentObj not in classesINeed:
			root.remove(obj)
			objNum  -= 1
		else:
			fileDict[currentObj].append(filename)
		
	if objNum  == 0 or filename[:-4] not in data_into_list:
		continue # drop the file, there are no objects of 'interest '
	else : # write to the file as xml to the new folder
		fwrite = open(modifiedXmlDir + filename , 'wb')
		print(fwrite)
		tree.write(fwrite)
		fwrite.close()

		# copy the corresponding JPEG to modifiedJPEGDIr
		copyfile(JPEGdirectory + filename[:-3] + 'jpg' , modifiedJPEGdir + filename[:-3] + 'jpg')
		imgnum += 1

		# make entry in the list file required for LMDB
		listfile.write(filename[:-4]+'\n')

		# Take only 101 images to train 
		# if imgnum == 101 :
		# 	break

	
	#print "found "+ str(objNum ) + " object(s) in " + filename[:-3]

listfile.close()
