import base64
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, make_response
from flask_bcrypt import Bcrypt
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
from io import BytesIO
import uuid
import json
import os
import pytz
from huggingface_hub import hf_hub_download
from xhtml2pdf import pisa
from db import users_collection, predictions_collection

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()





app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")


cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

ist = pytz.timezone('Asia/Kolkata')

bcrypt = Bcrypt(app)



REPO_ID = "Sharav1312/brain-tumor-models"

CLASS_NAME_MAP = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor"
}

DISEASE_INFO = {
    "Glioma": {
        "description": "Glioma is a tumor arising from glial cells in the brain.",
        "symptoms": "Headache, Seizures, Vision problems, Nausea",
        "treatment": "Surgery, Radiation therapy, Chemotherapy"
    },
    "Meningioma": {
        "description": "Meningioma develops in the protective membranes.",
        "symptoms": "Headache, Memory loss, Blurred vision",
        "treatment": "Surgery, Radiation therapy"
    },
    "Pituitary": {
        "description": "Pituitary tumor affects hormone regulation.",
        "symptoms": "Hormonal imbalance, Fatigue",
        "treatment": "Medication, Surgery"
    },
    "No Tumor": {
        "description": "No brain tumor detected.",
        "symptoms": "No tumor-related symptoms",
        "treatment": "No treatment required"
    }
}

#load model
def load_model(model_file):
    path = hf_hub_download(repo_id=REPO_ID, filename=model_file)
    return tf.keras.models.load_model(path)

def load_class_names():
    path = hf_hub_download(repo_id=REPO_ID, filename="class_names.json")
    with open(path, "r") as f:
        return json.load(f)

model_cnn = load_model("brain_tumor_model.keras")
model_vgg = load_model("brain_tumor_model_vgg16.keras")
class_names = load_class_names()

#image preprocessing
def predict_image(image, model, img_size=128):
    img = cv2.resize(image, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return np.argmax(preds), float(np.max(preds))


#routes

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        password = request.form['password']
        confirm_password = request.form['c_pass']

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        email = request.form['email']

        if bcrypt.check_password_hash(hashed_password, confirm_password):
            normal_user = {
            "user_id": str(uuid.uuid4()),
            "username": request.form['u_name'],
            "fullname": request.form['name'],
            "email": email,
            "phone" : request.form['phone'],
            "address" : request.form['address'],
            "gender" : request.form['gender'],
            "dob" : request.form['dob'],
            "age" : int(request.form['age']),
            "password": hashed_password,
            "created_at" :  datetime.now(ist).strftime("%d %b %Y, %I:%M %p")
            }

            existing_username = users_collection.find_one({"username" : normal_user['username']})

            if existing_username:
                flash("Username already exists")
                return render_template("register.html")
                
            users_collection.insert_one(normal_user)
            flash("Registered Sucessfully, It's Time to Log In", 'success')
            return redirect(url_for('login')) 
        
        else:
            flash("Password Doesn't match...Try again", 'danger')
            return render_template("register.html")
        
    
    return render_template("register.html")



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users_collection.find_one({"email": email})

        if user:
            check_password = bcrypt.check_password_hash(user['password'], password)

        if user and check_password:
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            session["fullname"] = user["fullname"]
            session["gender"] = user["gender"]
            session["age"] = user["age"]
            session["email"] = user["email"]
            session["phone"] = user["phone"]
            session["dob"] = user["dob"]
            session["address"] = user["address"]
            session['created_at'] = user["created_at"]
            session["logged_in_at"] = datetime.now(ist).strftime("%d %b %Y, %I:%M %p")

            flash("You've Successfully Loggged In", 'success')

            return redirect(url_for('user_profile'))
        
        else:
            flash("Invalid credentials !!!", 'error')

    return render_template("login.html")



@app.route("/user/predict", methods=["GET", "POST"])
def predict():

    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        if "image" not in request.files:
            return "No file part"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file"

        if file:

            file_bytes = file.read()

            if not file_bytes:
                return "Empty file uploaded"

            image = cv2.imdecode(
                np.frombuffer(file_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            if image is None:
                return "Invalid image file"

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            file.seek(0)

            upload_result = cloudinary.uploader.upload(file)

            image_url = upload_result["secure_url"]

            model_choice = request.form["model"]
            model = model_cnn if model_choice == "cnn" else model_vgg

            class_idx, confidence = predict_image(image_rgb, model)

            raw_label = class_names[class_idx].lower()
            label = CLASS_NAME_MAP.get(raw_label, raw_label)
            info = DISEASE_INFO[label]

            pred_id = str(uuid.uuid4())

            predictions_collection.insert_one({
                "pred_id": pred_id,
                "user_id": session['user_id'],
                "username": session["username"],
                "image_url": image_url,
                "prediction": label,
                "confidence": confidence,
                "predicted_at": datetime.now(ist).strftime("%d %b %Y, %I:%M %p")
            })

            return render_template(
                "predict.html",
                pred_id=pred_id,
                image_url=image_url,
                label=label,
                confidence=confidence,
                info=info
            )

    return render_template("predict.html")



@app.route("/report/<pred_id>")
def generate_report(pred_id):

    prediction = predictions_collection.find_one({"pred_id": pred_id})

    if not prediction:
        return "Prediction not found"

    user_id = prediction["user_id"]

    user = users_collection.find_one({"user_id": user_id})

    if not user:
        return "User not found"

    label = prediction["prediction"]
    confidence = float(prediction["confidence"])
    predicted_at = prediction["predicted_at"]

    logo_path = os.path.join("static", "images", "logo.jpg")

    if not os.path.exists(logo_path):
        return "Logo not found"

    with open(logo_path, "rb") as img:
        logo_base64 = base64.b64encode(img.read()).decode("utf-8")

    rendered = render_template(
        "pdf_report.html",
        label=label,
        confidence=confidence,
        predicted_at=predicted_at, 

        info=DISEASE_INFO[label],

        # User Data
        user=user,

        # System Data
        date_generated=datetime.now(),
        logo_base64=logo_base64
    )


    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(rendered, dest=pdf)

    if pisa_status.err:
        return "PDF generation failed"

    pdf.seek(0)

    safe_name = user["fullname"].replace(" ", "_")

    filename = f"TumorAI_Report_{safe_name}_{label}.pdf"

    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=False,     
        download_name=filename
    )


@app.route("/user/view_history")
def view_history():
    user_id = session['user_id']

    predictions = list(predictions_collection.find({"user_id": user_id}))

    
    return render_template("view_history.html", predictions = predictions)



@app.route("/logout")
def logout():
    session.clear()
    flash("You've Successfully Logged Out", 'success')
    return redirect(url_for("home"))


@app.route("/user/user_profile")
def user_profile():
    return render_template("user_profile.html")


if __name__ == "__main__":
    app.run(debug=True)