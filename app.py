from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import base64
import io
import matplotlib.pyplot as plt
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image
from sklearn.cluster import KMeans
import shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True 

#knn image retrivel
from sklearn.neighbors.unsupervised import NearestNeighbors
from PIL import Image
from tensorflow.keras.preprocessing import image



# Model saved with Keras model.save()
MODEL_PATH = 'models/garbage_model.h5'


# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# Check https://keras.io/applications/
#model = ResNet50(weights='imagenet')
#model.save('models')
#print('Model loaded. Check http://127.0.0.1:5000/')

# Variables
imdir = 'C:/Users/root/Documents/sample/deploy_sample/uploads/'
number_clusters = 3
filelist = glob.glob(os.path.join(imdir, '*.jpeg'))
#filelist = glob.glob(os.path.join(imdir ,'*.' + e)) for e in ext
    
# Define a flask app
app = Flask(__name__)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

#load images as np arrays
def load_data(fpath):    
    img=Image.open(fpath).resize((224,224)) # resize to 224x224 for training purposes
    img = np.asarray(img, dtype='float32')
    return img

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        output_class = ["batteries", "cloth", "e-waste", "glass", "light bulbs", "metallic", "organic", "paper", "plastic"]

        preds = model_predict(file_path, model)     
        print(preds)

        pred_class = output_class[np.argmax(preds)]
        pred_class_percent = round(np.max(preds) * 100, 2)

        result = 'It is '+ pred_class  + ' waste'            # Convert to string
        pred_class = ' with '+ str(pred_class_percent) + '% confidence'   
        
        #k-nn for recommending
        filelist.sort()
        featurelist = []
        for i, imagepath in enumerate(filelist):
                print("    Status: %s / %s" %(i, len(filelist)), end="\r")
                img = image.load_img(imagepath, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                features = np.array(model.predict(img_data))
                featurelist.append(features.flatten())
        nei_clf = NearestNeighbors(metric="euclidean")
        nei_clf.fit(featurelist)
        code = model_predict(file_path, model)
        (distances,),(idx,) = nei_clf.kneighbors(code,n_neighbors=3)
        

        #all images are loaded as np arrays
        images=[]
        labels=[]
        j=1
        for i,image_path in enumerate(filelist): 
            images.append(load_data(image_path))
        images = np.asarray(images) # all of the images are converted to np array of (1360,224,224,3)
        
        print(distances,images[idx])
        print(images[idx].shape)
        
        final_result =  result + pred_class 
        image_save = Image.fromarray((np.array(images[0]) * 255).astype(np.uint8))
        #image_save = Image.fromarray(images[idx], "RGB")
        image_save.save('out.jpg')
        image_output = os.path.join(basepath, 'out.jpg')
        immg = '<img src="'+image_output+'" style="height: 132px; width: 132px;">'
        #return render_template('index.html', filename=image_output)
        return final_result
    return None
    
@app.route('/recommend', methods=['POST'])
def recommend():
    #if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        #custer for recommending
        filelist.sort()
        featurelist = []
        for i, imagepath in enumerate(filelist):
                print("    Status: %s / %s" %(i, len(filelist)), end="\r")
                img = image.load_img(imagepath, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                features = np.array(model.predict(img_data))
                featurelist.append(features.flatten())
        nei_clf = NearestNeighbors(metric="euclidean")
        nei_clf.fit(featurelist)
        distances,neighbors = get_similar(file_path,n_neighbors=3)
        return 'hello recommender'

if __name__ == '__main__':
    app.run(debug=True)

