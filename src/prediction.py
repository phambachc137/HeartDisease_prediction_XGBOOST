# src/predict.py

import pandas as pd
from xgboost import XGBClassifier

from config import MODEL_PATH


def load_model():

#Load mô hình đã huấn luyện.
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def get_patient_information():
#Nhập thông tin bệnh nhân từ bàn phím.



    patient = {}

    patient["Age"] = int(input("Age: "))
    patient["Sex"] = int(input("Sex (0=Female, 1=Male): "))
    patient["RestingBP"] = int(input("RestingBP (80-200 mmHg): "))
    patient["Cholesterol"] = int(input("Cholesterol (mg/dL): "))
    patient["FastingBS"] = int(input("FastingBS (0<=120 mg/dL, 1>120 mg/dL): "))
    patient["MaxHR"] = int(input("MaxHR (60-220 bpm): "))
    patient["ExerciseAngina"] = int(input("ExerciseAngina (0=No, 1=Yes): "))
    patient["Oldpeak"] = float(input("Oldpeak (e.g. 0.0-6.5): "))

    patient["RestingECG_Normal"] = int(input("RestingECG_Normal (0=No, 1=Yes): "))
    patient["RestingECG_ST"] = int(input("RestingECG_ST (0=No, 1=Yes): "))

    patient["ST_Slope_Flat"] = int(input("ST_Slope_Flat (0=No, 1=Yes): "))
    patient["ST_Slope_Up"] = int(input("ST_Slope_Up (0=No, 1=Yes): "))

    patient["ChestPainType_ATA"] = int(input("ChestPainType_ATA (0=No, 1=Yes): "))
    patient["ChestPainType_NAP"] = int(input("ChestPainType_NAP (0=No, 1=Yes): "))
    patient["ChestPainType_TA"] = int(input("ChestPainType_TA (0=No, 1=Yes): "))
    
    return patient


def predict(patient):

    model = load_model()

    patient_df = pd.DataFrame([patient])

    prediction = model.predict(patient_df)[0]

    probability = model.predict_proba(patient_df)[0][1]

    print("\n========== RESULT ==========")

    if prediction == 1:
        print("Prediction : Heart Disease")
    else:
        print("Prediction : No Heart Disease")

    print(f"Probability : {probability*100:.2f}%")



def main():

    patient = get_patient_information()

    predict(patient)


if __name__ == "__main__":
    main()