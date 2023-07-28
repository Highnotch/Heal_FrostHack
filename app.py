from flask import Flask, render_template, request
from flask import  redirect, url_for
from PIL import Image
import werkzeug
from tensorflow.keras.preprocessing.image import  img_to_array
import cv2
import joblib
import os
import  numpy as np
import pickle
from tensorflow import keras
from tensorflow import keras
import tensorflow.keras.utils as utils
from keras.applications.vgg16 import preprocess_input

app= Flask(__name__)




@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home_stroke")
def home_stroke():
    return render_template("home_stroke.html")

@app.route("/home_heart")
def home_heart():
    return render_template("home_heart.html")

@app.route("/home_diabetes")
def home_diabetes():
    return render_template("home_diabetes.html")

@app.route("/home_kidney")
def home_kidney():
    return render_template("home_kidney.html")

@app.route("/home_liver")
def home_liver():
    return render_template("home_liver.html")

@app.route("/home_pneumonia")
def home_pneumonia():
    return render_template("home_pneumonia.html")

@app.route("/home_hepatitis")
def home_hepatitis():
    return render_template("home_hepatitis.html")

@app.route("/home_brain_tumor")
def home_brain_tumor():
    return render_template("home_brain_tumor.html")


@app.route("/brain_tumor_pred",methods=['POST','GET'])
def brain_tumor_pred():
    img = request.files['img']
    img.save('uploads/brain_tumor_img.jpg')

    image = Image.open("uploads/brain_tumor_img.jpg")
    model = keras.models.load_model('models/brain_tumor.h5',compile=(False))
    model.compile()
    x = np.array(image.resize((128,128)))
    x = x.reshape(1,128,128,3)
    res = model.predict_on_batch(x)
    pred = np.where(res == np.amax(res))[1][0]
    
    if pred==1:
        return render_template('no_disease.html',title='Brain Tumor')
    else:
        return render_template('yes_disease.html',title='Brain Tumor')


@app.route("/heart_pred",methods=['POST','GET'])
def heart_pred():
    age=int(request.form['age'])
    cholesterol=int(request.form['cholesterol'])
    fasting_blood_sugar=int(request.form['fasting_blood_sugar'])
    max_heart_rate_achieved = int(request.form['max_heart_rate_achieved'])
    exercise_induced_angina = int(request.form['exercise_induced_angina'])
    st_depression = float(request.form['st_depression'])
    chest_pain_type_typical_angina = int(request.form['chest_pain_type_typical angina'])
    rest_ecg_left_ventricular_hypertrophy = float(request.form['rest_ecg_left ventricular hypertrophy'])
    rest_ecg_normal = float(request.form['rest_ecg_normal'])
    st_slope_flat = int(request.form['st_slope_flat'])
    st_slope_upsloping = int(request.form['st_slope_upsloping'])

    x=np.array([age,cholesterol,fasting_blood_sugar,max_heart_rate_achieved,exercise_induced_angina,st_depression,
                chest_pain_type_typical_angina,rest_ecg_left_ventricular_hypertrophy,rest_ecg_normal,st_slope_flat,st_slope_upsloping]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_heart.pkl')
    scaler_heart=None
    with open(scaler_path,'rb') as scaler_file:
        scaler_heart=pickle.load(scaler_file)

    x=scaler_heart.transform(x)

    model_path=os.path.join('models/rf_ent_heart.sav')
    rf_ent=joblib.load(model_path)

    Y_pred=rf_ent.predict(x)

    # for No Heart Risk
    if Y_pred==0:
        return render_template('no_disease.html',title='Heart Disease')
    else:
        return render_template('yes_disease.html',title='Heart Disease')
#stroke

@app.route("/stroke_pred",methods=['POST','GET'])
def stroke_pred():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('models/dt.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return  render_template('no_disease.html',title='Stroke Risk')
    else:
        return render_template('yes_disease.html',title='Stroke Risk')

# diabetes
@app.route("/diabetes_pred",methods=['POST','GET'])
def diabetes_pred():
    
    Polyuria=int(request.form['Polyuria'])
    Polydipsia = int(request.form['Polydipsia'])
    age=int(request.form['age'])
    Gender=int(request.form['Gender'])
    partial_paresis	 = int(request.form['partial paresis'])
    sudden_wieght_loss = int(request.form['sudden wieght loss'])
    Irritability = int(request.form['Irritability'])
    delayed_healing	 = int(request.form['delayed healing'])
    Alopecia = int(request.form['Alopecia'])
    Itching = int(request.form['Itching'])
    
    x=np.array([Polyuria,Polydipsia,age,Gender,partial_paresis,sudden_wieght_loss,Irritability,delayed_healing,Alopecia,Itching]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_diabetes.pkl')
    scaler_diabetes=None
    with open(scaler_path,'rb') as scaler_file:
        scaler_diabetes=pickle.load(scaler_file)

    x=scaler_diabetes.transform(x)

    model_path=os.path.join('models/rf_diabetes.sav')
    rf_diabetes=joblib.load(model_path)

    Y_pred=rf_diabetes.predict(x)

    # for No Diabetes Risk
    if Y_pred==0:
        return render_template('no_disease.html',title='Diabetes Risk')
    else:
        return render_template('yes_disease.html',title='Diabetes Risk')

#kidney

@app.route("/kidney_pred",methods=['POST','GET'])
def kidney_pred():
    sg=float(request.form['sg'])
    al=float(request.form['al'])
    sc=float(request.form['sc'])
    hemo = float(request.form['hemo'])
    pcv = int(request.form['pcv'])
    htn = int(request.form['htn'])

    x=np.array([sg,al,sc,hemo,pcv,htn]).reshape(1,-1)
    print(x)
    scaler_path=os.path.join('models/scaler_kidney.pkl')
    scaler_kidney=None
    with open(scaler_path,'rb') as scaler_file:
        scaler_kidney=pickle.load(scaler_file)

    x=scaler_kidney.transform(x)

    model_path=os.path.join('models/rf_kidney.sav')
    rf_kidney=joblib.load(model_path)

    Y_pred=rf_kidney.predict(x)

    # for No ckd Risk
    if Y_pred==0:
        return render_template('no_disease.html',title='Kidney Disease')
    else:
        return render_template('yes_disease.html',title='Kidney Disease')

#liver
@app.route("/liver_pred",methods=['POST','GET'])
def liver_pred():
    age=int(request.form['age'])
    gender=int(request.form['gender'])
    Total_Bilirubin	=float(request.form['Total_Bilirubin'])
    Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
    Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
    Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
    Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
    Total_Protiens = float(request.form['Total_Protiens'])
    Albumin = float(request.form['Albumin'])
    Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])

    x=np.array([age,gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_liver.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('models/sv.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Liver Risk
    if Y_pred==0:
        return render_template('no_disease.html',title='Liver Disease')
    else:
        return render_template('yes_disease.html',title='Liver Disease')

#pneumonia
@app.route("/pneumonia_pred",methods=['POST','GET'])
def pneumonia_pred():
    
    img = request.files['img']
    img.save('uploads/pneumonia_img.jpeg')
    image = Image.open("uploads/pneumonia_img.jpeg")
    model = keras.models.load_model('models/chest_xray.h5',compile=False)
    model.compile()
    img1=utils.load_img("uploads/pneumonia_img.jpeg",target_size=(224,224))
    x=utils.img_to_array(img1)
    x=np.expand_dims(x, axis=0)
    img_data=preprocess_input(x)
    classes=model.predict(img_data)
    pred=int(classes[0][0])
    if pred==1:
        return render_template('no_disease.html',title='Pneumonia')
    else:
        return render_template('yes_disease.html',title='Pneumonia')


#brain tumor
@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    return 'bad request!', 400

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

if __name__=="__main__":
    app.run(debug=True,port=8000)