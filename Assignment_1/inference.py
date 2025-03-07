import joblib
import pandas as pd
import numpy as np

# Load models and encoders outside of the function
import os

base_dir = os.path.abspath("artifacts")
scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
encoder = joblib.load(os.path.join(base_dir, "encoder.pkl"))
best_model_knn = joblib.load(os.path.join(base_dir, "best_dt_model.pkl"))


def predict_outcomes(df):
    def categorize_bmi(bmi):
        if bmi < 16.0:
            return 'Severely_Underweight'
        elif bmi < 18.4:
            return 'Underweight'
        elif bmi < 24.9:
            return 'Normal'
        elif bmi < 29.9:
            return 'Overweight'
        elif bmi < 34.9:
            return 'Moderately_Obese'
        elif bmi < 40.0:
            return 'Severely_Obese'
        else:
            return 'Morbidly_Obese'

    # Categorize the BMI column
    df.loc[:,'BMI_Category'] = df['BMI'].apply(categorize_bmi)
    df=df.drop(columns=['BMI', 'Outcome'])

    # Define numeric and categorical features
    numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
    categorical_features = ['BMI_Category']

    # Scale numeric features
    df[numeric_features] = scaler.transform(df[numeric_features])

    # Encode categorical features
    encoded_df = pd.DataFrame(
        encoder.transform(df[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # Drop original categorical column
    df=df.drop(columns=categorical_features)

    # Combine scaled numeric and encoded categorical features
    input_final_df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Predict classes
    predicted_classes = best_model_knn.predict(input_final_df)

    return predicted_classes



