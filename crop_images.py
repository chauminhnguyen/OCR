import os
import numpy as np
import cv2
import pandas as pd
from google.colab.patches import cv2_imshow
import argparse


parser = argparse.ArgumentParser(description='Crop Images')
parser.add_argument('--csv', default='/content/data.csv', type=str, help='csv directory')
parser.add_argument('--test_folder', default='./test', type=str, help='text images directory')

args = parser.parse_args()

def crop(pts, image):

  """
  Takes inputs as 8 points
  and Returns cropped, masked image with a white background
  """
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  cropped = image[y:y+h, x:x+w].copy()
  pts = pts - pts.min(axis=0)
  mask = np.zeros(cropped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
  dst = cv2.bitwise_and(cropped, cropped, mask=mask)
  bg = np.ones_like(cropped, np.uint8)*255
  cv2.bitwise_not(bg,bg, mask=mask)
  dst2 = bg + dst

  return dst2


def generate_words(image_name, score_bbox, image):

  num_bboxes = len(score_bbox)
  it = 0
  for num in range(num_bboxes):
    bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
    if bbox_coords!=['{}']:
      l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0])
      t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1])
      r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0])
      t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1])
      r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
      b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
      l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
      b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']'))
      pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
      
      if np.all(pts) > 0:
        
        word = crop(pts, image)
        
        folder = '/'.join( image_name.split('/')[:-1])

        #CHANGE DIR
        dir = '/content/Result/Crop Words/'

        if os.path.isdir(os.path.join(dir + folder)) == False :
          os.makedirs(os.path.join(dir + folder))

        try:
          file_name = os.path.join(dir + image_name)
          cv2.imwrite(file_name+'_{}.jpg'.format(it), word)
          print('Image saved to '+file_name+'_{}.jpg'.format(it))
          it += 1
        except:
          continue

data=pd.read_csv(args.csv)

start = args.test_folder

for image_num in range(data.shape[0]):
  image = cv2.imread(os.path.join(start, data['image_name'][image_num]))
  image_name = data['image_name'][image_num].strip('.jpg')
  score_bbox = data['word_bboxes'][image_num].split('),')
  generate_words(image_name, score_bbox, image)