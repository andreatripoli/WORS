# The following project was implemented for the teaching project of Biometric Systems of the master's degree in
# Cybersecurity of Sapienza in Rome by the students Alessio Mobilia and Andrea Tripoli.

# The biometric system uses the face_recognition library for face recognition based on deep learning and the Gait
# Energy Image (GEI) method to preserve dynamic and static information of a gait sequence.

# Please refer to the README.md file for more information!



# Import the necessary packages
import imutils
import os
import numpy as np
import pickle
import time
import string
import datetime
import cv2
import csv
from PIL import Image, ImageQt
from pathlib import Path
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, \
    draw_bbox, read_class_names
from yolov3.configs import *
from detection.utils import checkFile, non_max_suppression_fast, calculateFrameForGei, get_gei_n_files, \
    load_dataset, save_gei, pickleFile, unpickleFile, recognize_face, set_label, print_label, checkDir
from collections import OrderedDict
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
import colorsys
import random
import face_recognition
import calendar

x1 =0
y1 = 0
x2 = 0
y2 = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class GeiPersonObj:
    def __init__(self,objectId,name, scoreface):
        self.objectId = objectId
        self.numInGEI = 0
        self.gei_current = np.zeros((128, 88), np.single)
        self.name = name
        self.scoreface = scoreface





def main():

    # String name
    pathImage = "report/img/"
    output = "output/"
    out=output
    input = "input/"
    encodingFace = 'data/encodingsFace.pickle'
    dataset_folder = "./data/"
    dataset_file_name = 'data'
    fileName = "report/"
    rep=fileName
    dirReport="report/"


    # Tolerance face recognition
    tolerance = 0.57
    tolerancetwo = 0.51

    # Da fare
    checkFile(input=input, output=output,
                            encodingFace= encodingFace)

    checkDir(dirReport)


    # Load the serialized dataset of faces
    print("[INFO] loading face encodings...")
    data = pickle.loads(open(encodingFace, "rb").read())

    yolo = Load_Yolo_model()

    for path, dirs, files in os.walk(input):
        for file in files:
            if file.endswith('.mp4'):

                # [IMPORTANT] Dictionary initialization to keep track of people who appeared in the video
                nameDetection = OrderedDict()
                timeDetection = OrderedDict()

                # Initialize the video stream and pointer to output video file
                print("[INFO] starting video stream...")
                print(os.path.join(path, file))
                capture = cv2.VideoCapture(os.path.join(path, file))
                output=out+file
                fileName = rep+str(os.path.splitext(file)[0])+'.csv'
                writer = None
                imgFaceFalse = cv2.imread("icon/faceFalse.jpg")
                imgGaitFalse = cv2.imread("icon/gaitFalse.jpg")
                imgFaceTrue = cv2.imread("icon/faceTrue.jpg")
                imgGaitTrue = cv2.imread("icon/gaitTrue.jpg")

                foregroundFaceFalse = imutils.resize(imgFaceFalse, width=20, height=20)
                foregroundFaceTrue = imutils.resize(imgFaceTrue, width=20, height=20)
                foregroundGaitFalse = imutils.resize(imgGaitFalse, width=20, height=20)
                foregroundGaitTrue = imutils.resize(imgGaitTrue, width=20, height=20)

                # Object tracker
                # Definition of the parameters
                max_cosine_distance = 0.7
                nn_budget = None

                # initialize deep sort object
                model_filename = 'model_data/mars-small128.pb'
                encoder = gdet.create_box_encoder(model_filename, batch_size=1)
                metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
                tracker = Tracker(metric)

                NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)
                key_list = list(NUM_CLASS.keys())
                val_list = list(NUM_CLASS.values())




                # Initialize for the gait
                firstFrame = None
                numInGEI = 0
                Face_identified = True
                max_n_gei = 200
                current_pickle_file = 0
                gei_current = np.zeros((128, 88), np.single)
                n_files = get_gei_n_files(dataset_folder, dataset_file_name)
                score_threshold = 0.90 #threshhold gait recognition
                gei_fix_num = 30
                num = 0
                gei = np.zeros([max_n_gei, 128, 88], np.uint8)
                name = []
                dataset_path = dataset_folder + dataset_file_name
                #id_name = "TEST"
                dataset_path, num, gei, name, loaded_dataset = load_dataset(dataset_path, num, gei, name, n_files)
                GeiList = []
                GeiUnknownList = []

                # Information
                countFrame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(capture.get(cv2.CAP_PROP_FPS))
                indexFrame = 0
                print("[INFO] start video generation ("+str(countFrame)+" frames)...")

                # Check if the file can be opened
                if not capture.isOpened():
                    print("[ERROR] the file cannot be opened...")
                    exit(0)

                # Open report file check
                with open(fileName, 'w') as f:

                    file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    file_writer.writerow(['Class', 'Identifier', 'name', 'Face image', 'Gait image', 'Video time'])

                    # Loop over frames from the video file stream
                    while True:

                        # Grab the frame from the threaded video stream
                        ret, frame = capture.read()

                        # If ret is True
                        if ret:

                            # Frame Information
                            indexFrame += 1
                            #print("[ACTIVITY] generation of "+str(indexFrame)+" in "+str(countFrame)+" frames...")

                            # Convert the input frame from BGR to RGB then resize it to have
                            # a width of 750px (to speedup processing)

                            

                            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                            original_frame = imutils.resize(original_frame, width=1200)
                            frame = imutils.resize(frame, width=1200)

                            # Apply background subtraction method.
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            gray = cv2.GaussianBlur(gray, (3, 3), 0)


                            # Set the first frame as background
                            if firstFrame is None:
                                firstFrame = gray
                            frameDelta = cv2.absdiff(firstFrame, gray)
                            thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]

                            thresh = cv2.dilate(thresh, None, iterations=2)
                            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            thresh = np.array(thresh)
                            max_rec = 0



                            image_data = image_preprocess(np.copy(original_frame), [416, 416])
                            # image_data = tf.expand_dims(image_data, 0)
                            image_data = image_data[np.newaxis, ...].astype(np.float32)

                            if YOLO_FRAMEWORK == "tf":
                                pred_bbox = yolo.predict(image_data)
                            elif YOLO_FRAMEWORK == "trt":
                                batched_input = tf.constant(image_data)
                                result = yolo(batched_input)
                                pred_bbox = []
                                for key, value in result.items():
                                    value = value.numpy()
                                    pred_bbox.append(value)

                            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                            pred_bbox = tf.concat(pred_bbox, axis=0)

                            bboxes = postprocess_boxes(pred_bbox, original_frame, 416, 0.3)
                            bboxes = nms(bboxes, 0.45, method='nms')

                            # extract bboxes to boxes (x, y, width, height), scores and names
                            boxes, scores, names = [], [], []

                            for bbox in bboxes:
                                if len(['person']) != 0 and NUM_CLASS[int(bbox[5])] in ['person'] or len(
                                        ['person']) == 0:
                                    boxes.append([bbox[0].astype(int), bbox[1].astype(int),
                                                  bbox[2].astype(int) - bbox[0].astype(int),
                                                  bbox[3].astype(int) - bbox[1].astype(int)])
                                    scores.append(bbox[4])
                                    names.append(NUM_CLASS[int(bbox[5])])

                            # Obtain all the detections for the given frame.
                            boxes = np.array(boxes)
                            names = np.array(names)
                            scores = np.array(scores)
                            features = np.array(encoder(original_frame, boxes))
                            detections = [Detection(bbox, score, class_name, feature) for
                                          bbox, score, class_name, feature in zip(boxes, scores, names, features)]

                            # Pass detections to the deepsort object and obtain the track information.
                            tracker.predict()
                            tracker.update(detections)

                            # Obtain info from the tracks
                            tracked_bboxes = []

                            for track in tracker.tracks:
                                if not track.is_confirmed() or track.time_since_update > 5:
                                    continue
                                bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                                class_name = track.get_class()  # Get the class name of particular object
                                tracking_id = track.track_id  # Get the ID for the particular track
                                index = key_list[
                                    val_list.index(class_name)]  # Get predicted object index by object name
                                tracked_bboxes.append(bbox.tolist() + [tracking_id,
                                                                       index])  # Structure data, that we could use it with our draw_bbox function



                            Text_colors = (255, 255, 0)
                            rectangle_colors = ''

                            NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)
                            num_classes = len(NUM_CLASS)
                            image_h, image_w, _ = original_frame.shape
                            hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
                            # print("hsv_tuples", hsv_tuples)
                            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

                            random.seed(0)
                            random.shuffle(colors)
                            random.seed(None)


                            # For each object recognized as a person
                            for i, bbox in enumerate(tracked_bboxes):
                                coor = np.array(bbox[:4], dtype=np.int32)
                                score = bbox[4]
                                class_ind = int(bbox[5])
                                bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
                                bbox_thick = int(0.6 * (image_h + image_w) / 1000)
                                if bbox_thick < 1: bbox_thick = 1
                                fontScale = 0.75 * bbox_thick
                                (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])



                                objectId = bbox[4]

                                arrayTime = [x1, x2, y1, y2, 0]

                                # Check if the person is stopped
                                if not objectId in timeDetection:
                                    timeDetection[objectId] = arrayTime
                                else:
                                    if (timeDetection[objectId][0] == arrayTime[0] and timeDetection[objectId][1] ==
                                            arrayTime[1]
                                            and timeDetection[objectId][2] == arrayTime[2] and timeDetection[objectId][
                                                3] == arrayTime[3]):
                                        if timeDetection[objectId][4] < 15:
                                            timeDetection[objectId][4] = timeDetection[objectId][4] + 1
                                        else:

                                            # deleting gei of old object no more needed
                                            for gl in GeiList:
                                                if gl.objectId == objectId:
                                                    GeiList.remove(gl)

                                            for gl in GeiUnknownList:
                                                if gl.objectId == objectId:
                                                    GeiUnknownList.remove(gl)
                                            break
                                    else:
                                        timeDetection[objectId] = arrayTime


                                try:
                                    # crop image based on some bounding box
                                    # rgb_cropped = frame[y1:y2, x1, x2]

                                    rgb = imutils.resize(original_frame, width=1200)
                                    subframe = rgb[y1:y2, x1:x2]

                                    boxes = face_recognition.face_locations(subframe,
                                                                            model="cnn")
                                    encodings = face_recognition.face_encodings(subframe, boxes)
                                    names = []
                                    scorL = []

                                    # loop over the facial embeddings
                                    for encoding in encodings:
                                        # attempt to match each face in the input image to our known
                                        # encodings
                                        # matches = face_recognition.compare_faces(data["encodings"],
                                        #                                         encoding, tolerance= x)

                                        

                                        face_distances = face_recognition.face_distance(data["encodings"],
                                                                                        encoding)

                                        nameP = "Unknown"
                                        score = 1
                                        scoref = 1

                                        matchedIdxs = [i for (i, b) in enumerate(face_distances) if b]
                                        for i in matchedIdxs:

                                            if (face_distances[i] < tolerance):
                                                # print(str(face_distances[i]))
                                                if (score > face_distances[i]):
                                                    score = face_distances[i]
                                                    nameP = data["names"][i]
                                                    scoref = score

                                        # update the list of names
                                        names.append(nameP)
                                        scorL.append(scoref)

                                    #if not objectId in nameDetection:
                                            #nameDetection[objectId] = ["person1", True, False, score face recognition]
                                    

                                    # loop over the recognized faces
                                    for ((top, right, bottom, left), nameT, valueScore) in zip(boxes, names, scorL):

                                        # rescale the face coordinates
                                        #cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), bbox_thick * 2)
                                        #cv2.putText(original_frame, name, (x1, y1 + 12), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        #            fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

                                        if not objectId in nameDetection:
                                            nameDetection[objectId] = [nameT, True, False, valueScore]

                                            if (nameT != "Unknown"):
                                                # Current GMT time in a tuple format
                                                current_GMT = time.gmtime()

                                                # ts stores timestamp
                                                ts = calendar.timegm(current_GMT)

                                                cv2.imwrite(
                                                    pathImage + str(os.path.splitext(file)[0]) + '(' + str(ts) + ').jpg',
                                                    original_frame)
                                                seconds = indexFrame / fps
                                                video_time = str(datetime.timedelta(seconds=seconds))

                                                file_writer.writerow(['Person', str(objectId), nameT,
                                                                      pathImage + str(
                                                                          os.path.splitext(file)[0]) + '(' + str(
                                                                          ts) + ').jpg', '',
                                                                      str(video_time)])



                                        else:
                                            if ((nameDetection[objectId][0] != nameT) and (nameT != "Unknown")):
                                                nameDetection[objectId][0] = nameT
                                                nameDetection[objectId][1] = True
                                                nameDetection[objectId][2] = False
                                                nameDetection[objectId][3] = valueScore


                                                # Current GMT time in a tuple format
                                                current_GMT = time.gmtime()

                                                # ts stores timestamp
                                                ts = calendar.timegm(current_GMT)

                                                cv2.imwrite(
                                                    pathImage + str(os.path.splitext(file)[0]) + '(' + str(ts) + ').jpg',
                                                    original_frame)
                                                seconds = indexFrame / fps
                                                video_time = str(datetime.timedelta(seconds=seconds))

                                                file_writer.writerow(['Person', str(objectId), nameT,
                                                                    pathImage + str(os.path.splitext(file)[0]) + '(' + str(ts) + ').jpg', '',
                                                                    str(video_time)])



                                            elif ((nameDetection[objectId][0] == nameT) and (nameT != "Unknown")):
                                                nameDetection[objectId][3] = valueScore

                                            else:
                                                nameDetection[objectId][1] = True


                                        if (nameDetection[objectId][0] == "Unknown"):
                                            nameDetection[objectId][1] = False




                                        

                                        


                                except Exception as e:
                                    #print(e)
                                    continue




                                # gait recognition
                                print(nameDetection)

                                # check if there are frame already saved for the person
                                GeiObj = next((x for x in GeiList if x.objectId == objectId), None)
                                if not objectId in nameDetection:
                                    detected_name = 'Unknown'
                                    face_detected = False
                                    scoreface = 1
                                else:
                                    detected_name = nameDetection[objectId][0]
                                    face_detected = nameDetection[objectId][1]
                                    scoreface = nameDetection[objectId][3]

                                if (GeiObj == None):
                                    GeiObj = GeiPersonObj(objectId, detected_name, scoreface)
                                    GeiList.append(GeiObj)
                                elif GeiObj.name != detected_name:
                                    GeiObj.name = detected_name
                                    GeiObj.scoreface = scoreface


                                

                                # check if there are some gei for the same person that before has not been recognized
                                if (detected_name != 'Unknown' and face_detected and scoreface < tolerancetwo):
                                    for ug in GeiUnknownList:
                                        if (
                                                ug.objectId == objectId):  # check if it is in the unknown list and remove it
                                            ug.name = detected_name
                                            GeiUnknownList.remove(ug)
                                            if (ug.numInGEI > gei_fix_num):  # this should be true ever

                                                # to save the gei in the last file is needed to load the last file
                                                n_files = get_gei_n_files(dataset_folder, dataset_file_name)
                                                if loaded_dataset != n_files:  # load the next database file
                                                    dataset_path, num, gei, name, loaded_dataset = load_dataset(
                                                        dataset_path, num, gei,
                                                        name, n_files)

                                                # save the gei of people that have been recorded and now are known
                                                gei, ug.gei_current, gei_fix_num, dataset_path, num, ug.numInGEI, current_pickle_file = save_gei(
                                                    gei,
                                                    ug.gei_current, gei_fix_num, dataset_path, num, ug.name,
                                                    ug.numInGEI, current_pickle_file, name, max_n_gei)


                                            else:
                                                # append the gei to the GEI list of known people
                                                GeiList.append(ug)

                                if (x2 - x1) < (y2 - y1):

                                    try:

                                        nim = np.zeros([thresh.shape[0] + 10, thresh.shape[1] + 10],
                                                        np.single)  # Enlarge the box for better result
                                        nim[y1 + 10:(y1 + (y2 - y1) + 10), x1 + 10:(x1 + (x2 - x1) + 10)] = thresh[y1:(
                                                    y1 + (y2 - y1)), x1:(x1 + (x2 - x1))]
                                        # Get coordinate position.
                                        ty, tx = (nim > 100).nonzero()




                                        sy, ey = ty.min(), ty.max() + 1
                                        sx, ex = tx.min(), tx.max() + 1
                                        h = ey - sy
                                        w = ex - sx

                                        GeiObj.numInGEI, GeiObj.gei_current = calculateFrameForGei(tx=tx, h=h, w=w,
                                                                                                    sx=sx, ex=ex,
                                                                                                    sy=sy, ey=ey,
                                                                                                    nim=nim,
                                                                                                    numInGEI=GeiObj.numInGEI,
                                                                                                    gei_current=GeiObj.gei_current,
                                                                                                    gei_fix_num=gei_fix_num)

                                        if GeiObj.numInGEI > gei_fix_num:

                                            # if it has been identified save the gait
                                            if GeiObj.name != "Unknown" and face_detected and GeiObj.scoreface < tolerancetwo:

                                                # check if exist the same gei in the database
                                                exist = False
                                                gei_query = GeiObj.gei_current / (gei_fix_num)
                                                n_files = get_gei_n_files(dataset_folder, dataset_file_name)
                                                for nf in range(n_files + 1):  # loop all the database files
                                                    if (not exist):
                                                        if loaded_dataset != nf:  # load the next database file
                                                            dataset_path, num, gei, name, loaded_dataset = load_dataset(
                                                                dataset_path, num, gei,
                                                                name, nf)
                                                        score = 0
                                                        if num >= 0:
                                                            for q in range(
                                                                    num):  # loop all the gei in the single database file
                                                                gei_to_com = gei[q, :, :]
                                                                score = np.exp(-(((gei_query[:] - gei_to_com[:]) / (
                                                                        128 * 88)) ** 2).sum())  # Compare with gait database (return a number between 0 and 1)
                                                                if (
                                                                        score >= 0.99):  # check if the score is above the threshold
                                                                    exist = True
                                                                    break
                                                    else:
                                                        break

                                                # save the gei if it is not already saved
                                                if (not exist):
                                                    # save the gei
                                                    gei, GeiObj.gei_current, gei_fix_num, dataset_path, num, GeiObj.numInGEI, current_pickle_file = save_gei(
                                                        gei,
                                                        GeiObj.gei_current, gei_fix_num, dataset_path, num,
                                                        GeiObj.name,
                                                        GeiObj.numInGEI, current_pickle_file, name, max_n_gei)

                                                GeiList.remove(GeiObj)
                                                del GeiObj


                                            elif GeiObj.name == "Unknown":
                                                #print("step 1")

                                                # Recognition.
                                                gei_query = GeiObj.gei_current / (gei_fix_num)

                                                gei_to_com = np.zeros([128, 88], np.single)
                                                # count the files
                                                n_files = get_gei_n_files(dataset_folder, dataset_file_name)
                                                detected = False
                                                q_id = 0
                                                if (n_files >= 0):
                                                    #print("step 2")

                                                    max_scores = np.zeros(n_files + 1)
                                                    max_scores_index = np.zeros(n_files + 1)
                                                    for nf in range(n_files + 1):  # loop all the database files

                                                        if loaded_dataset != nf:  # load the next database file
                                                            dataset_path, num, gei, name, loaded_dataset = load_dataset(
                                                                dataset_path, num, gei,
                                                                name, nf)

                                                        score = np.zeros(num)

                                                        if num > 0:
                                                            for q in range(
                                                                    num):  # loop all the gei in the single database file
                                                                gei_to_com = gei[q, :, :]
                                                                score[q] = np.exp(
                                                                    -(((gei_query[:] - gei_to_com[:]) / (
                                                                            128 * 88)) ** 2).sum())  # Compare with gait database (return a number between 0 and 1)
                                                            q_id = score.argmax()
                                                            if (score[
                                                                q_id] >= score_threshold):  # check if the score is above the threshold
                                                                detected = True

                                                            # save the max score in the checked database file
                                                            max_scores[nf] = score[q_id]
                                                            max_scores_index[nf] = q_id

                                                            #print(str(score[q_id]))

                                                if detected:


                                                    # take the max score between all the max scores
                                                    nf_index = max_scores.argmax()
                                                    q_id = int(max_scores_index[nf_index])

                                                    # load the data in the dataset file in which is the maxscore
                                                    if loaded_dataset != nf_index:
                                                        dataset_path, num, gei, name, loaded_dataset = load_dataset(
                                                            dataset_path, num, gei,
                                                            name, nf_index)

                                                    id_rec = '%s' % name[q_id]
                                                    GeiObj.name = name[q_id]
                                                    #print('gei confidence' + str(max_scores[nf_index]))
                                                    # remove to the GEI list
                                                    GeiList.remove(GeiObj)

                                                    if not objectId in nameDetection:
                                                        nameDetection[objectId] = [GeiObj.name, False, True, 1]
                                                    elif nameDetection[objectId][1] == False:
                                                        nameDetection[objectId] = [GeiObj.name, False, True, 1]

                                                    # add detection entry in the csv file
                                                    seconds = indexFrame / fps
                                                    video_time = str(datetime.timedelta(seconds=seconds))
                                                    file_writer.writerow(['Person', str(objectId), GeiObj.name, '',
                                                                            'dataset/' + str(
                                                                                GeiObj.name) + '/gei/' + str(
                                                                                nf_index) + '_' + str(q_id),
                                                                            str(video_time)])




                                                else:
                                                    # add to the GEI list of Unknowns
                                                    GeiUnknownList.append(GeiObj)
                                                    GeiList.remove(GeiObj)


                                    except Exception as e :
                                        print("Persona ignorata")


                                # end gait






                                print_label(original_frame, x1, y1, x2, y2, foregroundFaceFalse, foregroundGaitFalse,
                                            foregroundFaceTrue, foregroundGaitTrue, nameDetection, objectId)

                            # If the video write is None
                            if writer is None:
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(output, fourcc, fps,
                                                        (original_frame.shape[1], original_frame.shape[0]), True)

                            # If the write is not None, write the frame with recognized faces to disk
                            if writer is not None:
                                writer.write(original_frame)

                            original_frame = imutils.resize(original_frame, width=1200)
                            cv2.imshow("Frame", original_frame)
                            key = cv2.waitKey(1) & 0xFF
                            # if the `q` key was pressed, break from the loop
                            if key == ord("q"):
                                break

                        else:
                            print("[INFO] generation almost over...")
                            break

                    print("[INFO] cleanup...")
                    # Cleanup
                    cv2.destroyAllWindows()
                    capture.release()

                    # Check to see if the video write point needs to be released
                    if writer is not None:
                        writer.release()

                    print("[INFO] all done...")

if __name__ == "__main__":
    main()
