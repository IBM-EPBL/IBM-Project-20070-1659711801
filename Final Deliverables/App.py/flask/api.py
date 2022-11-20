import numpy as np #used for numerical analysis
from flask import Flask, render_template, request #Flask is a application
import tensorflow
#used to run/serve our aplication
#request is used to access the file which is uploaded by the user in #our application
#render_template is used for rendering the html pages
from tensorflow.keras.models import load_model #ve are Loading our model from keras
app = Flask (__name__) #our flask app
model = load_model('crude_oil.h5',) #loading the model in the flask app
@app.route('/',methods = ['POST','GET']) #rendering html template
def home():
  return render_template("home.html") #rendering html template
@app.route('/predict',methods = ['POST','GET'])
def predict():
  return render_template("predict.html") #rendering html template
@app.route('/login', methods = ['POST','GET']) #route for our prediction 
def login():
  x_input=str(request.form['year']) #requesting the file
  x_input=x_input.split(',')
  print(x_input)
  for i in range(0, len(x_input)):
     x_input[i]=float(x_input[i])
  print(x_input)
  x_input=np.array(x_input).reshape(1,-1)
  temp_input=list(x_input)
  temp_input=temp_input[0].tolist()
  lst_output=[]
  n_steps=10
  i=0
  while(i<1):
    if(len(temp_input)>10):
#print("temp input", temp_input)
      x_input=np.array(temp_input[1:])
      print("{} day input {}".format(i,x_input))
      x_input=x_input.reshape(1,-1)
      x_input = x_input.reshape((1, n_steps, 1))
#print(x_input)
      yhat=model.predict(x_input, verbose=0)
      print("{} day output {}".format(i,yhat))
      temp_input.extend(yhat[0].tolist())
      temp_input=temp_input[1:]
#print (temp_input)
      lst_output.extend(yhat.tolist())
      i=i+1

    else:
      x_input = x_input.reshape((1, n_steps,1)) 
      yhat=model.predict(x_input, verbose=0)
      print(yhat[0])
      temp_input.extend(yhat[0].tolist())
      print(len(temp_input))
      lst_output.extend(yhat.tolist())
      i=i+1
      print(lst_output)
  return render_template("predict.html", showcase = 'The next day predicted value is: '+str(lst_output))
#return str(x)
if __name__=='__main__':
   app.run(debug=True, port=5000)
