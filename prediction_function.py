
def predict_insurance_charges(age, sex, bmi, children, smoker, region):
    """
    Predict insurance charges for a person

    Args:
        age (int): Age of the person
        sex (str): 'male' or 'female'
        bmi (float): Body Mass Index
        children (int): Number of children
        smoker (str): 'yes' or 'no'
        region (str): 'northeast', 'northwest', 'southeast', 'southwest'

    Returns:
        float: Predicted insurance charges
    """
    import joblib
    import pandas as pd
    import numpy as np

    # Load model artifacts
    artifacts = joblib.load('insurance_prediction_model.pkl')
    model = artifacts['model']
    label_encoders = artifacts['label_encoders']
    feature_names = artifacts['feature_names']

    # Create input data
    person_data = {
        'age': age, 'sex': sex, 'bmi': bmi,
        'children': children, 'smoker': smoker, 'region': region
    }

    # Preprocessing (same as training)
    person_df = pd.DataFrame([person_data])
    person_df['sex_encoded'] = label_encoders['sex'].transform(person_df['sex'])
    person_df['smoker_encoded'] = label_encoders['smoker'].transform(person_df['smoker'])
    person_df['region_encoded'] = label_encoders['region'].transform(person_df['region'])

    # Feature engineering
    person_df['age_squared'] = person_df['age'] ** 2
    person_df['bmi_squared'] = person_df['bmi'] ** 2
    person_df['age_bmi_interaction'] = person_df['age'] * person_df['bmi']
    person_df['smoker_age_interaction'] = person_df['smoker_encoded'] * person_df['age']
    person_df['smoker_bmi_interaction'] = person_df['smoker_encoded'] * person_df['bmi']
    person_df['is_obese'] = (person_df['bmi'] >= 30).astype(int)

    # Categories and dummies (simplified for deployment)
    bmi_cats = ['bmi_Normal', 'bmi_Obese', 'bmi_Overweight', 'bmi_Underweight']
    age_cats = ['age_Middle_Age', 'age_Pre_Senior', 'age_Senior', 'age_Young', 'age_Young_Adult']

    for cat in bmi_cats + age_cats:
        person_df[cat] = 0

    # Set appropriate categories
    if bmi < 18.5:
        person_df['bmi_Underweight'] = 1
    elif bmi < 25:
        person_df['bmi_Normal'] = 1
    elif bmi < 30:
        person_df['bmi_Overweight'] = 1
    else:
        person_df['bmi_Obese'] = 1

    if age < 25:
        person_df['age_Young'] = 1
    elif age < 35:
        person_df['age_Young_Adult'] = 1
    elif age < 45:
        person_df['age_Middle_Age'] = 1
    elif age < 55:
        person_df['age_Pre_Senior'] = 1
    else:
        person_df['age_Senior'] = 1

    # Make prediction
    X_pred = person_df[feature_names]
    prediction = model.predict(X_pred)[0]

    return round(prediction, 2)

# Example usage:
# cost = predict_insurance_charges(30, 'male', 25.0, 1, 'no', 'northeast')
# print(f"Predicted cost: ${cost:,.2f}")
