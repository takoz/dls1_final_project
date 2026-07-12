import os
import io
import streamlit as st

from pathlib import Path
from ultralytics import YOLO

import cv2
import numpy as np
import pandas as pd


yolo26s_obb_summary = '''
```
YOLO26s-obb summary (fused): 132 layers, 9,753,489 parameters, 0 gradients, 21.5 GFLOPs

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        366        416      0.969      0.907      0.946      0.773
        elbow positive         62         68      0.952      0.873      0.909      0.762
      fingers positive         88        107      0.953      0.925      0.955      0.717
      forearm fracture         64         70      0.984      0.901      0.951       0.77
      humerus fracture         57         58      0.976      0.983      0.983      0.827
     shoulder fracture         60         67      0.993      0.955      0.993      0.856
        wrist positive         35         46      0.957      0.804      0.883      0.708
```
'''

yolo26n_obb_summary = '''
```
YOLO26n-obb summary (fused): 132 layers, 2,447,577 parameters, 0 gradients, 5.4 GFLOPs

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        366        416      0.902      0.784      0.885      0.584
        elbow positive         62         68      0.936      0.858      0.938      0.617
      fingers positive         88        107       0.82      0.683      0.806      0.458
      forearm fracture         64         70      0.951      0.836      0.885      0.593
      humerus fracture         57         58      0.981      0.895      0.968      0.705
     shoulder fracture         60         67       0.86      0.826      0.954      0.649
        wrist positive         35         46      0.862      0.609      0.759      0.484
```
'''


model_to_weights = {
  'yolo26s-obb': {
    'weights': os.path.join('res', 'models', 'yolo26s-obb.pt'),
    'summary': yolo26s_obb_summary
  },
  'yolo26n-obb': {
    'weights': os.path.join('res', 'models', 'yolo26n-obb.pt'),
    'summary': yolo26n_obb_summary
  }
}


def make_inference(model_path, image, conf, iou):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    image_array = opencv_image[..., ::-1]

    model = YOLO(model_path)
    results = model.predict(source=image_array, conf=conf, iou=iou, save=False)
    img = results[0].plot()
    return results, img


def showcase(model, conf, iou):
 samples_dir = Path('res/samples/clean')
 gt_dir = Path('res/samples/gt')
 samples = samples_dir.glob('*.jpg')
 sample_files = []
 sample_names = []
 gt_images = []

 for sample in samples:
   with open(sample, 'rb') as f:
     sample_files.append(io.BytesIO(f.read()))
   sample_names.append(sample.name)

   gt_image = os.path.join(gt_dir, sample.name)
   with open(gt_image, 'rb') as f:
     gt_images.append(io.BytesIO(f.read()))

 predict(model, sample_files, sample_names, conf, iou, gt_images)


def predict(model, images, image_names, conf, iou, gt_images=None):
   for i, image in enumerate(images):
     out, out_img = make_inference(model, image, conf, iou)

     st.divider()

     st.write(
       f'Stats: preprocess {out[0].speed['preprocess']:.2f} ms, '
       f'inference {out[0].speed['inference']:.2f} ms, '
       f'postprocess {out[0].speed['postprocess']:.2f} ms, '
     )

     col1, col2, col3 = st.columns(3)

     col1.image(image, caption='Original', channels='RGB')
     col2.image(out_img, caption='Predicted', channels='RGB')

     if gt_images is not None:
      col3.image(gt_images[i], caption='Ground truth', channels='RGB')


def app():
  st.title('Bone Fracture Detection')
  st.set_page_config(layout="wide")

  sideb = st.sidebar

  conf_slider = sideb.slider("Confidence Threshold", 0.0, 1.0, 0.25)
  iou_slider = sideb.slider("IoU Threshold", 0.0, 1.0, 0.7)

  uploaded_files = sideb.file_uploader(
    'Choose an image file',
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
  )

  model = sideb.selectbox(
    'Model',
    ['yolo26s-obb', 'yolo26n-obb']
  )

  st.markdown(model_to_weights[model]['summary'])

  predict_button = sideb.button('Predict')
  showcase_button = sideb.button('Demo',type="primary")

  if predict_button:
    if len(uploaded_files) == 0:
      st.write('No images provided')
    else:
      image_names = [image.name for image in uploaded_files]
      predict(model_to_weights[model]['weights'], uploaded_files,
              image_names, conf_slider, iou_slider)

  if showcase_button:
    showcase(model_to_weights[model]['weights'], conf_slider, iou_slider)


if __name__ =='__main__':
  app()
