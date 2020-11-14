from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import walk

import numpy as np
from PIL import Image
import tensorflow as tf # TF2

dd = "dinar2000"
pp = "C:/Users/Admin/Desktop/test/"+dd+"/"
rr = 0
nn = 0
list1 = []
model_file='C:/Users/Admin/Desktop/IMG_DALLA/PFE_APP/app/android/mlkit-automl/app/src/main/assets/automl/model.tflite'
label_file='tmp/labels.txt'
input_mean=127.5
input_std=127.5


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
  f = []
  for (dirpath, dirnames, filenames) in walk(pp):
    f.extend(filenames)
    break
  
  image=f

  interpreter = tf.lite.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  for im in image:
      nn += 1
      img = Image.open(pp+im).resize((width, height))
      input_data = np.expand_dims(img, axis=0)

      if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
      interpreter.set_tensor(input_details[0]['index'], input_data)
    

      interpreter.invoke()

    
      output_data = interpreter.get_tensor(output_details[0]['index'])
      results = np.squeeze(output_data)
    
      top_k = results.argsort()[-5:][::-1]
      labels = load_labels(label_file)
      for i in top_k:
          print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
          while float(results[i] / 255.0) > 0.5 :
              list1.append(labels[i])       
              break
          
    
      print("-----")
      
      
  for i in list1:
      if i == dd:
          rr += 1    
  print(str(rr)+" out of "+str(nn)+" are correct")
  print(str(100-(nn*rr/100))+"% correct")
