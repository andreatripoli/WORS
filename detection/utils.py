# General functions for recognition and screen prints

# Import the necessary packages
from genericpath import isdir
import os.path
import pickle
import numpy as np
import face_recognition
import cv2
import string
import random
import csv
import datetime
import shutil
from PIL import Image, ImageQt
from pathlib import Path

# Initialization
max_n_gei = 200
offsetX = 20

# Check for existence of files
def checkFile(input, output, encodingFace):
    if os.path.isdir(input):
        if os.path.isfile(encodingFace):
            if not os.path.isdir(output):
                print("[ERROR] there is no output folder")
                exit(0)
            inputfiles=False
            for root, dirs, files in os.walk(input):
                for file in files:
                    if file.endswith('.mp4'):
                        inputfiles=True
                        if(os.path.isfile(output+file)):
                            try:
                                os.remove(output+file)
                            except:
                                print("[ERROR] impossible remove files in output")
                                exit(0)

            if not inputfiles:
                print("[ERROR] there are no input files")
                exit(0)



        else:
            print("[ERROR] the encodingsFace file does not exist...")
            exit(0)
    else:
        print("[ERROR] the input directory does not exist...")
        exit(0)
    return True


# Function to identify the coordinates of the class
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))




def pickleFile(filename, data, compress=False):
    fo = open(filename, "wb")
    pickle.dump(data, fo, protocol=pickle.HIGHEST_PROTOCOL)
    fo.close()


def unpickleFile(filename):
    fo = open(filename, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict



#Load gait database if exists
def load_dataset(dataset_path, num, gei, name, current_pickle_file):
    if(current_pickle_file == -1):
        current_pickle_file = 0
    path=dataset_path+str(current_pickle_file)
    if os.path.exists(path): #take variable frome file
        dic = unpickleFile(path)
        num = dic['num']
        gei = dic['gei']
        name = dic['name']
        #print("dataset loaded: ",current_pickle_file)
        loaded_dataset = current_pickle_file
    else: #create new file
        num = 0
        gei = np.zeros([max_n_gei,128,88],np.uint8)
        name = []
        dic = {'num':num, 'gei':gei, 'name':name}
        pickleFile(path, dic, compress=False)
        print('created new dataset called: ',path)
        loaded_dataset = current_pickle_file
    return dataset_path, num, gei, name, loaded_dataset



#Save the GEI.
def save_gei(gei, gei_current, gei_fix_num, dataset_path, num, id_name, numInGEI,current_pickle_file, name, max_n_gei):
    
    # check if the face should be saved in a new file
    if (num >= max_n_gei):
        current_pickle_file += 1
        num = 0

    path=dataset_path+str(current_pickle_file)
    gei[num,:,:] = gei_current/gei_fix_num
    Path("./dataset/"+id_name).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(gei_current/gei_fix_num)).save('./dataset/%s/gei%d_%d.jpg'%(id_name,current_pickle_file,num))
    name.append(id_name)
    num +=1
    #id_num.setText('%d' %num)
    dic = {'num':num, 'gei':gei, 'name':name}
    pickleFile(path, dic, compress=False)
    print('GEI Saved!')
    gei_current = np.zeros((128,88), np.single)
    numInGEI = 0
    return gei, gei_current, gei_fix_num, dataset_path, num, numInGEI, current_pickle_file



#return the number of gei files-1 (the number in the name of the last file)
def get_gei_n_files(dataset_folder,dataset_file_name):
    totalFiles = -1

    entries = Path(dataset_folder)
    for entry in entries.iterdir():
            if entry.name.startswith(dataset_file_name):
                totalFiles += 1


    return totalFiles




# Calculate the frame for GEI
def calculateFrameForGei(tx, h, w, sx, ex, sy, ey, nim, numInGEI, gei_current, gei_fix_num):
    cx = int(tx.mean())
    cenX = h / 2
    start_w = (h - w) / 2
    if max(cx - sx, ex - cx) < cenX:
        start_w = cenX - (cx - sx)
    tim = np.zeros((h, h), np.single)
    tim[:, int(start_w):int(start_w + w)] = nim[int(sy):int(ey), int(sx):int(ex)]
    rim = Image.fromarray(np.uint8(tim)).resize((128, 128), Image.ANTIALIAS)
    tim = np.array(rim)[:, offsetX:offsetX + 88]
    if numInGEI < gei_fix_num:
        gei_current += tim  # Add up until reaching the fix number.
    numInGEI += 1

    return numInGEI, gei_current



def recognize_face(rgb, data, tolerance):

    try:

        boxes = face_recognition.face_locations(rgb, model="cnn")

        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:

            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding, tolerance)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        return boxes, names

    except:
        boxes = []
        names = []

        return boxes, names


def set_label(boxes, names, frame, r, objectId, dic, file, x1, x2, y1, y2, foregroundFaceFalse, foregroundGaitFalse,
                foregroundFaceTrue, foregroundGaitTrue, indexframe, fps):
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):

        if not objectId in dic:
            dic[objectId] = [None, False, False]
            dic[objectId][0] = name
        else:
            if (dic[objectId][0] == name) and (dic[objectId][2] == True):
                dic[objectId][2] = True
            else:
                dic[objectId][0] = name
                dic[objectId][2] = False


        if (name != "Unknown"):
            rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            dic[objectId][1] = True
            seconds = int(indexframe / fps)
            video_time = str(datetime.timedelta(seconds=seconds))
            cv2.imwrite('report/img/' + str(objectId) + '(' + str(rand) + ').jpg', frame)
            file.writerow(['Person', str(objectId), name, 'report/img/' + str(objectId) + '(' + str(rand) + ').jpg', '', str(video_time)])







def print_label(frame, x1, y1, x2, y2, foregroundFaceFalse, foregroundGaitFalse,
                foregroundFaceTrue, foregroundGaitTrue, dic, objectId):


    if(objectId in  dic):
        if(dic[objectId][1] or dic[objectId][2]):

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            

            if(dic[objectId][1]):
                # Select the region in the background where we want to add the image and add the images
                # using cv2.addWeighted()
                added_image = cv2.addWeighted(frame[(y1 - 20):y1, x1:(x1 + 20), :], 0,
                                  foregroundFaceTrue[0:20, 0:20, :], 1, 0)
                
            else:
                added_image = cv2.addWeighted(frame[(y1 - 20):y1, x1:(x1 + 20), :], 0,
                                  foregroundFaceFalse[0:20, 0:20, :], 1, 0)
            # Change the region with the result
            frame[(y1 - 20):y1, x1:(x1 + 20)] = added_image



            if(dic[objectId][2]):
                # Select the region in the background where we want to add the image and add the images
                # using cv2.addWeighted()
                added_image2 = cv2.addWeighted(frame[(y1 - 20):y1, (x1 + 25):(x1 + 45), :], 0,
                                            foregroundGaitTrue[0:20, 0:20, :], 1, 0)

            else:
                added_image2 = cv2.addWeighted(frame[(y1 - 20):y1, (x1 + 25):(x1 + 45), :], 0,
                                            foregroundGaitFalse[0:20, 0:20, :], 1, 0)
            # Change the region with the result
            frame[(y1 - 20):y1, (x1 + 25):(x1 + 45)] = added_image2


            cv2.putText(frame, dic[objectId][0], (x1 + 48, y1 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (100, 30, 250), 1)

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


def checkDir(dirReport):
    try:
        if (os.path.isdir(dirReport)):
            shutil.rmtree(dirReport, ignore_errors=True)
            os.mkdir(dirReport)
            os.mkdir(dirReport+'img/')
        else:
            os.mkdir(dirReport)
            os.mkdir(dirReport+'img/')
    except:
        print('[ERROR]cannot delete the directory')
        exit(0)
    return True




