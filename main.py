from flask import Flask, request, render_template, jsonify
#from pythonapp.forms import ContactForm
#import imageio
#from scipy.misc.pilutil import imsave, imread, imresize
#import keras.models
import pickle
import numpy as np

#from scipy.misc import imsave, imread, imresize
#import keras.models
#import re
#import sys
#import os
#import base64
#sys.path.append(os.path.abspath("./models"))
#from load import *

#global graph, models

#model, graph = init()


app = Flask(__name__)

#def convertImage(imgData1):
#	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
#	with open('output.png','wb') as output:
#	    output.write(base64.b64decode(imgstr))

#@app.route('/predictskin/',methods=['GET','POST'])
#def predictskin():
#	imgData = request.get_data()
#	convertImage(imgData)
#	x = imread('output.png',mode='L')
#	x = np.invert(x)
#	x = imresize(x,(28,28))
#	x = x.reshape(1,28,28,1)
#
#	with graph.as_default():
#		out = model.predict(x)
#		print(out)
#		print(np.argmax(out,axis=1))
#
#		response = np.array_str(np.argmax(out,axis=1))
#		return response



##################################################

#import sklearn
#from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#from keras.models import Sequential, load_model
#from keras.layers import Dense
#import keras as k





model = pickle.load(open('model.pkl', 'rb'))
modeldiab=pickle.load(open('modeldiab.pkl', 'rb'))
modelkid = pickle.load(open('modelkid.pkl', 'rb'))
modelbreast = pickle.load(open('modelbreast.pkl', 'rb'))
modellung = pickle.load(open('modellung.pkl', 'rb'))
modelliv = pickle.load(open('modelliv.pkl', 'rb'))


@app.route("/")
@app.route("/index.html")

#@app.route("/", methods=['POST','GET'])
#@app.route("/index.html", methods=['POST','GET'])
def home():
    #form = ContactForm()
    #if request.method=='POST':
        #name=form.name.data
        #email=form.email.data
        #message=form.message.data
        #print(name, email, message)
        #return render_template('index.html', form=form)
    #return render_template('index.html', form=form)
    return render_template('index.html')




@app.route("/connect.html")
def connect():
    return render_template('connect.html')

@app.route("/community.html")
def community():
    return render_template('community.html')

@app.route("/share.html")
def share():
    return render_template('share.html')

@app.route("/read.html")
def read():
    return render_template('read.html')

@app.route("/ask.html")
def ask():
    return render_template('ask.html')


########################################################################################################################################################



@app.route("/diagheart.html")
def heart():
    return render_template('diagheart.html')


@app.route("/diagdiab.html")
def diab():
    return render_template('diagdiab.html')


@app.route("/diagbreast.html")
def breast():
    return render_template('diagbreast.html')


@app.route("/diaglung.html")
def lung():
    return render_template('diaglung.html')


@app.route("/diagskin.html")
def skin():
    return render_template('diagskin.html')


@app.route("/diagbrain.html")
def brain():
    return render_template('diagbrain.html')


@app.route("/diagkidney.html")
def kid():
    return render_template('diagkidney.html')


@app.route("/diagpm.html")
def pm():
    return render_template('diagpm.html')


@app.route("/diagal.html")
def al():
    return render_template('diagal.html')


@app.route("/diagthy.html")
def thy():
    return render_template('diagthy.html')


@app.route("/diagliv.html")
def liv():
    return render_template('diagliv.html')

#######################################################################################################################################################


@app.route("/dietheart.html")
def heart1():
    return render_template('dietheart.html')


@app.route("/dietdiab.html")
def diab1():
    return render_template('dietdiab.html')


@app.route("/dietbreast.html")
def breast1():
    return render_template('dietbreast.html')


@app.route("/dietlung.html")
def lung1():
    return render_template('dietlung.html')


@app.route("/dietskin.html")
def skin1():
    return render_template('dietskin.html')


@app.route("/dietbrain.html")
def brain1():
    return render_template('dietbrain.html')


@app.route("/dietkidney.html")
def kid1():
    return render_template('dietkidney.html')


@app.route("/dietpn.html")
def pn1():
    return render_template('dietpn.html')


@app.route("/dietal.html")
def al1():
    return render_template('dietal.html')


@app.route("/dietpar.html")
def par1():
    return render_template('dietpar.html')


@app.route("/dietthy.html")
def thy1():
    return render_template('dietthy.html')


@app.route("/diethep.html")
def hep1():
    return render_template('diethep.html')


@app.route("/dietliver.html")
def liv1():
    return render_template('dietliver.html')


@app.route("/dietchol.html")
def chol1():
    return render_template('dietchol.html')


@app.route("/dietbp.html")
def bp1():
    return render_template('dietbp.html')



#######################################################################################################################################################




@app.route("/exercisehead.html")
def head():
    return render_template('exercisehead.html')

@app.route("/exerciseuter.html")
def uter():
    return render_template('exerciseuter.html')

@app.route("/exerciseheart.html")
def heart2():
    return render_template('exerciseheart.html')


@app.route("/exercisediab.html")
def diab2():
    return render_template('exercisediab.html')


@app.route("/exercisebreast.html")
def breast2():
    return render_template('exercisebreast.html')


@app.route("/exerciselung.html")
def lung2():
    return render_template('exerciselung.html')


@app.route("/exercisebrain.html")
def brain2():
    return render_template('exercisebrain.html')


@app.route("/exercisechol.html")
def chol2():
    return render_template('exercisechol.html')


@app.route("/exercisekidney.html")
def kid2():
    return render_template('exercisekidney.html')


@app.route("/exercisepm.html")
def pn2():
    return render_template('exercisepm.html')


@app.route("/exerciseal.html")
def al2():
    return render_template('exerciseal.html')


@app.route("/exercisepar.html")
def par2():
    return render_template('exercisepar.html')


@app.route("/exercisethy.html")
def thy2():
    return render_template('exercisethy.html')


@app.route("/exercisehep.html")
def hep2():
    return render_template('exercisehep.html')


@app.route("/exerciseliv.html")
def liv2():
    return render_template('exerciseliv.html')


@app.route("/exercisebp.html")
def bp2():
    return render_template('exercisebp.html')

@app.route("/resheart.html")
def resheart():
    return render_template('resheart.html')

@app.route("/result.html")
def result():
    return render_template('result.html')

@app.route("/resbr.html")
def resbr():
    return render_template('resbr.html')

@app.route("/reskid.html")
def reskid():
    return render_template('reskid.html')

@app.route("/resliv.html")
def resliv():
    return render_template('resliv.html')

####################################################################################################################################################

@app.route('/heartpredict', methods=['POST'])
def heartpredict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
      ch = "Low :)"
    else:
      ch = "High :("
    return render_template('/resheart.html', prediction_text=' Result: ' +'Risk Level is '+ ch)

#****************************************************************************************************************
@app.route('/diabpredict', methods=['POST'])
def diabpredict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = modeldiab.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
      ch = "Low :)"
    else:
      ch = "High :("
    return render_template('/result.html', prediction_text='Result: ' +'Risk Level is '+ ch)

#******************************************************************************************************************
@app.route('/predictbr', methods=['POST'])
def predictbr():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = modelbreast.predict(final_features)
    output = prediction[0]
    if output == 'B':
      ch = "Benign :)"
    else:
      ch = "Malignant :("
    return render_template('/resbr.html', prediction_text=' Result: ' +'The tumor is '+ ch)
#******************************************************************************************************************
@app.route('/predictlung', methods=['POST'])
def predictlung():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = modellung.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
      ch = "Low :)"
    else:
      ch = "High :("
    return render_template('/reslung.html', prediction_text='Result: ' +'Risk Level is '+ ch)

#******************************************************************************************************************

@app.route('/predictkid', methods=['POST'])
def predictkid():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = modelkid.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
      ch = "Low :)"
    else:
      ch = "High :( "
    return render_template('/reskid.html', prediction_text=' Result: ' +'Risk Level is '+ ch)
#******************************************************************************************************************

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)
	x = imread('output.png',mode='L')
	x = np.invert(x)
	x = imresize(x,(28,28))
	x = x.reshape(1,28,28,1)

	with graph.as_default():
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))

		response = np.array_str(np.argmax(out,axis=1))
		return response


#******************************************************************************************************************
#@app.route('/predictliv', methods=['POST'])
#def predictliv():
#    # For rendering results on HTML GUI
#    int_features = [float(x) for x in request.form.values()]
#    final_features = [np.array(int_features)]
#    prediction = modelliv.predict(final_features)
#    output = round(prediction[0], 2)
#    if output == 2:
#      ch = "Low :)"
#    else:
#      ch = "High :("
#    return render_template('/resliv.html', prediction_text=' Result: ' +'Risk Level is '+ ch)
#******************************************************************************************************************
#******************************************************************************************************************
#******************************************************************************************************************
if __name__ == "__main__":
    app.run(debug=True)
