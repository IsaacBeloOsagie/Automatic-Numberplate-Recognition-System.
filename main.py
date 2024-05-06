from flask import Flask, request, jsonify
from flask_cors import CORS
from AnprsService import AnprsService
from CnnWrapper import CnnWrapper

app = Flask(__name__)
CORS(app)

@app.route("/")
def home() : 
    return "Services Is Up and Running"

#Normal Web Page. Not a REST Endpoint 
@app.route("/cnn-process-image", methods=["POST"])
def cnnImageProcessor():
   file_image = request.files['image'];
   CnnWrapperRunner = CnnWrapper(file_image)
   CnnWrapperRunner.execute()


# REST Endpoint - Controller 
@app.route("/process-image", methods=["POST"])
def OpticalImageProcessor() :
     # 1. Read and Decode Image from request object - Posted From REACT
    file_image = request.files['image'];

    # 2. Process the Image
    anprs_service = AnprsService(file_image)
    response = anprs_service.process_image() # upto step 3b is working fine.

    # 3. Return the Encode response to REACT frontend
    return jsonify(response), 200

if __name__ == "__main__" : 
    app.run(debug=True)