from flask import Flask,request,json
import os
from flask_cors import CORS
from helper import initialize
from helper import predict_image_class

global mod
app =Flask(__name__)
cors = CORS(app)


@app.route("/")
def home():
    return json.jsonify({
    "username": "admin",
    "email": "admin@localhost",
    "id": 42
})

@app.route("/getimg",methods=['POST'])
def upload_image():
    mod=initialize()
    print(request.files['imagefile'])
    imgfile=request.files['imagefile']
    imagePath="./images/"+imgfile.filename
    imgfile.save(imagePath)
    probabilities, predicted_class_index, predicted_class_name = predict_image_class(imagePath,mod)
    return {"Probability": str(max(probabilities)),
            "PredictedClassName": predicted_class_name}

if __name__=="__main_":
    
    app.run()