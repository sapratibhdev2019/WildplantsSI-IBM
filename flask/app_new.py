from tensorflow.keras.layers import LayerNormalization 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications.vgg16 import preprocess_input 
from tensorflow.keras.applications.vgg16 import VGG16 
#import required libraries 
import numpy as np 
import os 
from PIL import Image
 
#import Flask 
from flask import Flask , request, render_template 
from werkzeug.utils import secure_filename 
from gevent.pywsgi import WSGIServer 
 
#import keras 
#import keras 
from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing import image 
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam 
 
#graph = tf.get_default_graph() 
 
 
app = Flask(__name__) 
model = load_model("WildPlantEdibility.h5") 
 
@app.route('/') 
def index(): 
    return render_template('index.html') 
 
@app.route('/predict',methods = ['GET','POST']) 
def upload(): 
    if request.method == 'POST': 
        f = request.files['image'] 
        print("current path") 
        basepath = os.path.dirname(__file__) 
        print("current path", basepath) 
        filepath = os.path.join(basepath,'uploads',f.filename) 
        print("upload folder is ", filepath) 
        f.save(filepath) 
         
        img = image.load_img(filepath,target_size = (224, 224)) 
        x = image.img_to_array(img) 
        x = np.expand_dims(x,axis =0) 
        #IMAGE_SIZE = [224, 224] 
        #x = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) 
        #rint(x) 
        img_data=preprocess_input(x) 
        print(img_data)
         
        a=model.predict(img_data) 
        print(a)
        preds=np.argmax(a, axis=1) 
 
        #with graph.as_default(): 
        #preds = model.predict(x) 
        print("prediction",preds) 
             
        index=['Mountain Laurel_nonedible', 'Peppergrass_edible', 'Purple Deadnettle_edible',  
       'Rhododendron_nonedible', 'Toothwort_edible', 'Wild Grape Vine_edible', 'Wild Leek_edible', 'rattlebox_nonedible'] 
        text = "prediction : "+ index[preds[0]] 
        print(text) 
        return text 
    return render_template("index.html", z = text) 
     
if __name__ == '__main__': 
    app.run(debug = True)

