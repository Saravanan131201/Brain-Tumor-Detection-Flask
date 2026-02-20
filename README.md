# 🧠 Brain Tumor Detection System

An AI-powered web application for detecting brain tumors from MRI images using Deep Learning.  
Built with Flask, TensorFlow/Keras, MongoDB, and Cloudinary.

---

## 🚀 Features

✔ Upload MRI brain scans  
✔ CNN / VGG16 model selection  
✔ Tumor prediction with confidence score  
✔ Prediction history dashboard  
✔ AI-generated PDF reports  
✔ User authentication (Register / Login)  
✔ Cloud-based image storage (Cloudinary)  
✔ MongoDB database integration  

---

## 🛠️ Tech Stack

### Frontend
- HTML  
- CSS  
- Bootstrap  

### Backend
- Flask  
- Python  

### Machine Learning
- TensorFlow / Keras  
- Custom CNN  
- VGG16 (Transfer Learning)  

### Database
- MongoDB Atlas  

### Cloud Services
- Cloudinary (Image Storage)  

---

## 📸 Application Workflow

1. User registers / logs in  
2. Uploads MRI scan  
3. Selects prediction model  
4. AI analyzes image  
5. Displays:
   - Diagnosis  
   - Confidence Score  
   - Medical Information  

6. Prediction stored in MongoDB  
7. PDF report available for download  

---

## 🧠 Models Used

### ✅ Custom CNN
- Lightweight  
- Faster inference  

### ✅ VGG16 (Transfer Learning)
- Higher accuracy  
- Deep feature extraction  

---

## ⚙️ Environment Variables (.env)

Create a `.env` file in the root directory:

SECRET_KEY=your_secret_key

MONGO_URI=your_mongodb_connection_string

CLOUDINARY_CLOUD_NAME=your_cloud_name  
CLOUDINARY_API_KEY=your_api_key  
CLOUDINARY_API_SECRET=your_api_secret

---


### 3️⃣ Install Dependencies

pip install -r requirements.txt

---

### 4️⃣ Run Application

python app.py

---

## 🌐 Deployment (Render)

✔ Gunicorn WSGI server  
✔ Environment variables configured  
✔ Cloudinary for persistent image storage  
✔ MongoDB Atlas cloud database  

---

## 📊 Prediction Output

The system provides:

- Tumor classification  
- Confidence percentage  
- Clinical-style PDF report  

---

## ⚠️ Disclaimer

This application is intended for **educational and research purposes only**.

❗ Not a substitute for professional medical diagnosis  
❗ Always consult a qualified medical professional  

---

