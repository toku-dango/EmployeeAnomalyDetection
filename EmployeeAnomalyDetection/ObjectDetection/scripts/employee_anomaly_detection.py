# coding: utf-8
# Object Detection Demo
import argparse
import cv2
import numpy as np
import os
import sys
import time
import datetime #<<<<<<必要なモジュールのimportを追加
import subprocess #<<<<<<必要なモジュールのimportを追加
import requests #<<<<<<必要なモジュールのimportを追加
import tensorflow as tf

from distutils.version import StrictVersion

try:
  if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
except:
  pass

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='object detection tester, using webcam or movie file')
parser.add_argument('-l', '--labels', default='./models/coco-labels-paper.txt', help="default: './models/coco-labels-paper.txt'")
parser.add_argument('-m', '--model', default='./models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb', help="default: './models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'") #<<<<<<実行時に引数を少なくするために変更
parser.add_argument('-d', '--device', default='raspi_cam', help="normal_cam, jetson_nano_raspi_cam, jetson_nano_web_cam, raspi_cam, or video_file. default: 'raspi_cam'") # normal_cam / jetson_nano_raspi_cam / jetson_nano_web_cam 
parser.add_argument('-i', '--input_video_file', default='', help="Input video file")

args = parser.parse_args()

detection_graph = tf.Graph()

mode = 'bbox'

colors = [
  (0, 0, 255),
  (0, 64, 255),
  (0, 128, 255),
  (0, 192, 255),
  (0, 255, 255),
  (0, 255, 192),
  (0, 255, 128),
  (0, 255, 64),
  (0, 255, 0),
  (64, 255, 0),
  (128, 255, 0),
  (192, 255, 0),
  (255, 255, 0),
  (255, 192, 0),
  (255, 128, 0),
  (255, 64, 0),
  (255, 0, 0),
  (255, 0, 64),
  (255, 0, 128),
  (255, 0, 192),
  (255, 0, 255),
  (192, 0, 255),
  (128, 0, 255),
  (64, 0, 255),
]

#__________追加__________
BUILDING_NAME = "A" #建物番号
ROOM_NUM = "202" #部屋番号

#capacityJudge
CaBefTimes=[0,0,0,0,0,0]
CAPACITY_NUM = 3 #<<<<<<密集検知 しきい値 (3人以上で密集検知カウンターを上げる)
CaJudgeCounter = 0 #<<<<<<密集検知 カウンター
CaResetTime = 20 #<<<<<<密集検知カウンター リセット時間
CaAlertThreshold = 60 #<<<<<<密集検知 カウンター しきい値

#emergencyJudge
MovBefTimes=[0,0,0,0,0,0]
MovJudgeCounter = 0 #<<<<<<異常検知カウンター
MovResetTime = 30 #<<<<<<異常検知カウンター リセット時間
MovAlertThreshold = 60 #<<<<<<異常検知 カウンター しきい値
MovAlertWaitTime = 180 #<<<<<<通知待機時間設定
#画像の格納パス
pictpath='/home/pi/Desktop/CheckView/ObjectDetection/picts' #<<<<<<画像を保存するパス

#サーバ通知間隔
interval=300

#moveJudge
moveThreshold=1000 #<<<<<<動体検知 しきい値
avg=None


def moveDetect(img, avg):
    moveFlg = 0
    #画像をグレースケールに変換
    grayImg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #前回画像がない場合の処理
    if avg is None:
        avg = grayImg.copy().astype("float")
        return avg, moveFlg
 
    #前画像との差分を取得する
    cv2.accumulateWeighted(grayImg, avg, 0.00001)
    delta = cv2.absdiff(grayImg, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(delta, 50, 255, cv2.THRESH_BINARY)[1]
    contours, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    #画像内 差分箇所のうち最大箇所を抽出
    max_area=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if max_area < area:
            max_area = area;
     
    #動体判定
    if max_area > moveThreshold:
        moveFlg = 1

    #今回取得画像を保存
    avg = grayImg.copy().astype("float")

    return avg, moveFlg

def emergencyJudge(img, nowTime, MovJudgeCounter):

    if MovJudgeCounter == 0:
        MovJudgeCounter = 1
        #初回検知時間を保存
        MovBefTimes[0]=int(nowTime)
    else:
        if int(nowTime) - MovBefTimes[1] < MovResetTime:
             MovJudgeCounter += 1
        else:
            MovJudgeCounter = 0
    print("MoveCounter=", MovJudgeCounter)
    #検知時間を保存
    MovBefTimes[1]=int(nowTime)
        
    #通知(しきい値及び経過時間が規定以上になった場合に通知)
    if MovJudgeCounter >= MovAlertThreshold and int(nowTime) - MovBefTimes[0] > MovAlertWaitTime:
        if int(nowTime) - MovBefTimes[2] > interval:
            #通知時間を保存
            MovBefTimes[1]=int(nowTime)
            #ファイルに保存 
            filename=pictpath+'/'+nowstr+'_EmergencyCall.png'
            cv2.imwrite(filename, img)
            messageLine = "従業員異常通知_会議室" + BUILDING_NAME + ROOM_NUM
            messageAlexa = BUILDING_NAME + ROOM_NUM + "従業員の異常を確認しました。状況を確認してください。"
            lineMessage(filename, messageLine)
            alexaMessage(messageAlexa)
            sendSoracom(filename)
                
    return MovJudgeCounter

def capacityJudge(personNum, img, nowTime, CaJudgeCounter):
    if personNum >= (CAPACITY_NUM / 2):
        
        if CaJudgeCounter == 0:
            CaJudgeCounter = 1
        else:
            if int(nowTime) - CaBefTimes[0] < CaResetTime:
                CaJudgeCounter += 1
            else:
                CaJudgeCounter = 0
        #検知時間を保存
        CaBefTimes[0]=int(nowTime)
        print("CapacityCounter=", CaJudgeCounter)
            
        #通知(一定時間間隔)
        if CaJudgeCounter >= CaAlertThreshold:
            if int(nowTime) - CaBefTimes[1] > interval:
                #通知時間を保存
                CaBefTimes[1]=int(nowTime)
                #ファイルに保存 
                filename=pictpath+'/'+nowstr+'MaxCapacity.png'
                cv2.imwrite(filename, img)
                
                messageLine = "密集防止アラート_会議室" + BUILDING_NAME + ROOM_NUM
                messageAlexa = BUILDING_NAME + ROOM_NUM + "会議室の収容人数を超過しています。注意喚起を実施ください。"
                lineMessage(filename, messageLine)
                alexaMessage(messageAlexa)
                sendSoracom(filename)

    return CaJudgeCounter

def lineMessage(fname, message):
    url = "https://notify-api.line.me/api/notify"
    token = [""]#ここにLINE Notifyのトークンを入力
    for i in token:
        headers = {"Authorization" : "Bearer "+ i}
        payload = {"message" :  message}
        files = {"imageFile": open(fname, "rb")}
        r = requests.post(url, headers = headers, params=payload, files=files)
        print(r.text)


def alexaMessage(message):
    message = "speak:" + message
    cmd = ["./alexa_remote_control.sh", "-e", message]
    res = subprocess.call(cmd)

def sendSoracom(filename):
    send_imageFile ='curl -v -X PUT --data-binary @'  + filename + ' -H content-type:images/png http://harvest-files.soracom.io/lagoon/harvest.png'
    send_message ='curl -v -X POST -H content-type:application/json -d {\"latitude\":34.6872318,\"longitude\":135.5259173,\"room_num\":' + ROOM_NUM + '} http://harvest.soracom.io'
    #<<<<<<緯度経度/部屋番号情報を送る(緯度経度は、サンプルとして大阪城の位置情報となっています)
    res = subprocess.call(send_message.split())
    res = subprocess.call(send_imageFile.split())

def load_graph():
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x_min, y_min, x_max, y_max, ratio=0.1):
    dst = src.copy()
    dst[y_min:y_max, x_min:x_max] = mosaic(dst[y_min:y_max, x_min:x_max], ratio)
    return dst

# Load a (frozen) Tensorflow model into memory.
print('Loading graph...')
detection_graph = load_graph()
print('Graph is loaded')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with detection_graph.as_default():
  tf_sess = tf.Session(config = tf_config)
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)

  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

def run_inference_for_single_image(image, graph):
  # Run inference
  output_dict = tf_sess.run(tensor_dict,
                          feed_dict={image_tensor: image})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

if args.input_video_file != "":
  # WORKAROUND
  print("[Info] --input_video_file has an argument. so --device was replaced to 'video_file'.")
  args.device = "video_file"

# Switch camera according to device
if args.device == 'normal_cam':
  cam = cv2.VideoCapture(0)
elif args.device == 'jetson_nano_raspi_cam':
  GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx \
    ! videoconvert \
    ! appsink drop=true sync=false'
  cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
elif args.device == 'jetson_nano_web_cam':
  cam = cv2.VideoCapture(1)
elif args.device == 'raspi_cam':
  from picamera.array import PiRGBArray
  from picamera import PiCamera
  cam = PiCamera()
  cam.resolution = (640, 480)
  stream = PiRGBArray(cam)
elif args.device == 'video_file':
  cam = cv2.VideoCapture(args.input_video_file)
else:
  print('[Error] --device: wrong device')
  parser.print_help()
  sys.exit()

count_max = 0

if __name__ == '__main__':
  count = 0

  labels = ['blank']
  with open(args.labels,'r') as f:
    for line in f:
      labels.append(line.rstrip())

  while True:
    if args.device == 'raspi_cam':
      cam.capture(stream, 'bgr', use_video_port=True)
      img = stream.array
    else:
      ret, img = cam.read()
      if not ret:
        print('error')
        break

    key = cv2.waitKey(1)
    if key == 77 or key == 109: # when m or M key is pressed, go to mosaic mode
      mode = 'mosaic'
    elif key == 66 or key == 98: # when b or B key is pressed, go to bbox mode
      mode = 'bbox'
    elif key == 27: # when ESC key is pressed break
        break

    count += 1
    if count > count_max:
      img_bgr = cv2.resize(img, (300, 300))

      # convert bgr to rgb
      image_np = img_bgr[:,:,::-1]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      start = time.time()
      output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
      elapsed_time = time.time() - start
      
      befDetectList = []#<<<<<追加する 検出した物体名を格納するリストの定義
      personNum = 0 #<<<<<追加する 検出人数を0で置く
      avg, moveFlg = moveDetect(img, avg)#動体検知関数

      for i in range(output_dict['num_detections']):
        class_id = output_dict['detection_classes'][i]
        if class_id < len(labels):
          label = labels[class_id]
        else:
          label = 'unknown'
          
        befDetectList.append(label)#<<<<<追加する 検出した物体名をリストに格納

        detection_score = output_dict['detection_scores'][i]

        if detection_score > 0.5:
            # Define bounding box
            h, w, c = img.shape
            box = output_dict['detection_boxes'][i] * np.array( \
              [h, w,  h, w])
            box = box.astype(np.int)

            speed_info = '%s: %.3f' % ('fps', 1.0/elapsed_time)
            #cv2.putText(img, speed_info, (10,50), \
            #  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            if mode == 'bbox':
              class_id = class_id % len(colors)
              color = colors[class_id]

              # Draw bounding box
              cv2.rectangle(img, \
                (box[1], box[0]), (box[3], box[2]), color, 3)

              # Put label near bounding box
              information = '%s: %.1f%%' % (label, output_dict['detection_scores'][i] * 100.0)
              cv2.putText(img, information, (box[1] + 15, box[2] - 15), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
            elif mode == 'mosaic':
              img = mosaic_area(img, box[1], box[0], box[3], box[2], ratio=0.05)
            
            if label ==  "person": #<<<<<追加する
              personNum += 1  #<<<<<追加する 検出した人物数を数える

      #現在日付を取得
      nowstr=datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #<<<<<追加する
      nowTime=time.time() #<<<<<追加する

      if moveFlg == 0 and personNum == 1: #<<<<<追加する
          MovJudgeCounter = emergencyJudge(img, nowTime, MovJudgeCounter) #<<<<<追加する 異常検知
      CaJudgeCounter = capacityJudge(personNum, img, nowTime, CaJudgeCounter) #<<<<<追加する 密集検知
      
      cv2.putText(img, "Capacity_Judge:"+str(CaJudgeCounter) , (250,30), \
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
      cv2.putText(img, "Emergency_Judge:"+str(MovJudgeCounter) , (400,30), \
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
      
      cv2.imshow('detection result', img)
      count = 0
            
    if args.device == 'raspi_cam':
      stream.seek(0)
      stream.truncate()

  tf_sess.close()
  cam.release()
  cv2.destroyAllWindows()
