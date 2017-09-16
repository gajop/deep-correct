import os
import xml.etree.ElementTree as ET
import glob
import shutil
import random

INPUT_DIR = "../data/pascal/VOCdevkit/VOC2012/"
IMAGE_DIR = os.path.join(INPUT_DIR, "JPEGImages")
ANNOTATION_DIR = os.path.join(INPUT_DIR, "Annotations")
DEST_DIR = "../data/pascal/processed/"

def get_xml_name(xmlFile):
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    return root.find("object").find("name").text

def copy_file(filePath, subdir, className):
    fileName = os.path.basename(filePath)
    destPath = os.path.join(DEST_DIR, subdir, className, fileName)
    if os.path.exists(destPath):
        return

    destSubDir = os.path.join(DEST_DIR, subdir, className)
    if not os.path.exists(destSubDir):
        print("Making destination subdir: {}".format(destSubDir))
        os.makedirs(destSubDir)

    shutil.copyfile(filePath, destPath)

def countClasses():
    classes = {}
    for i, filePath in enumerate(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
        fileName = os.path.basename(filePath)
        fileBase = fileName[:-4]
        annotationPath = os.path.join(ANNOTATION_DIR, fileBase + ".xml")
        try:
            className = get_xml_name(annotationPath)
            if className not in classes:
                classes[className] = 0
            classes[className] += 1
        except Exception as ex:
            print(ex)
            continue

    return classes

def get_max_class_count():
    classCounts = countClasses()
    return min([v for k, v in classCounts.iteritems()]) * 4

def copy_all_files():
    train_val_map = get_train_val_sets()

    classCount = {}
    classLimit = get_max_class_count()
    copiedCount = 0
    for i, filePath in enumerate(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
        fileName = os.path.basename(filePath)
        fileBase = fileName[:-4]
        annotationPath = os.path.join(ANNOTATION_DIR, fileBase + ".xml")
        try:
            className = get_xml_name(annotationPath)
        except:
            print("Errored with: {}".format(annotationPath))
            continue
        # if fileBase not in train_val_map:
        #     print("File not in either train or val: {}".format(fileBase))
        #     continue
        # subdir = train_val_map[fileBase]

        if className not in classCount:
            classCount[className] = 0
        if classCount[className] > classLimit and False:
            continue

        classCount[className] += 1


        rnd = random.random()
        if rnd > 0.8:
#            subdir = 'd3'
#        elif rnd > 0.4:
            subdir = 'd2'
        else:
            subdir = 'd1'
        copy_file(filePath, subdir, className)
        copiedCount = copiedCount + 1
    print("Copied: {}".format(copiedCount))

def get_train_val_sets():
    train_list = open(os.path.join(INPUT_DIR, "ImageSets", "Main", "train.txt")).readlines()
    val_list = open(os.path.join(INPUT_DIR, "ImageSets", "Main", "val.txt")).readlines()

    train_val_map = {}
    for fname in train_list:
        train_val_map[fname[:-1]] = "train"
    for fname in val_list:
        train_val_map[fname[:-1]] = "val"

    return train_val_map

copy_all_files()
#countClasses()
#get_train_val_sets()
