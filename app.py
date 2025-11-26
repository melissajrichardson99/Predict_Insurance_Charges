import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from PIL import Image

model = load_model('dt_insurance_charges')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data= input_df)
    predictions = predictions_df.iloc[0]['prediction_label']
    return predictions
def run():
    image = Image.open('image2.png')
    image_hospital = Image.open('image.jpg')

    st.image(image)
    st.sidebar.info("This app is created to predict patient hospital insurance charges")
    st.sidebar.image(image_hospital)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.title("Insurance Charges Prediction App")
    if add_selectbox == 'Online':
        age = st.number_input('Age', min_value = 0, max_value = 150, value = 25)
        gender = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value = 10, max_value = 50, value = 10)
        children = st.selectbox('Number of Children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker = 'Yes'
        else:
            smoker = 'No'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])
        output = ""

        input_dict = {'age': age, 'sex': gender, 'bmi': bmi, 'children':children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model = model, input_df = input_df)
            output = "$"+str(output)
        st.success('This Output is {}'.format(output))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload a csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data = data)
            st.write(predictions)


if __name__ == '__main__':
    run()