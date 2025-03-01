import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\toxic_vect.pkl', 'rb') as f:
    toxic = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\toxic_model.pkl', 'rb') as f:
    toxic_model = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\severe_toxic_vect.pkl', 'rb') as f:
    severe_toxic = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\severe_toxic_model.pkl', 'rb') as f:
    severe_toxic_model = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\threat_vect.pkl', 'rb') as f:
    threat = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\threat_model.pkl', 'rb') as f:
    threat_model = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\obscene_vect.pkl', 'rb') as f:
    obscene = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\obscene_model.pkl', 'rb') as f:
    obscene_model = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\insult_vect.pkl', 'rb') as f:
    insult = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\insult_model.pkl', 'rb') as f:
    insult_model = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\identity_hate_vect.pkl', 'rb') as f:
    identity_hate = pickle.load(f)
with open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\identity_hate_model.pkl', 'rb') as f:
    identity_hate_model = pickle.load(f)

def main():
    st.title("Toxic Comments Classification")
    image = Image.open(r'c:\Users\Hanisha\Desktop\Data_Science__Machine_Learning-Projects-main\Project 2\Toxic-Comment-Classification\Image.jpg')
    st.image(image, use_column_width=True)

    Input = st.text_input("Don't Hold Back - Be as rude as you can [in english only]")
    if len(Input) == 0:
        return
    
    vect = severe_toxic.transform([Input])
    zero = severe_toxic_model.predict_proba(vect)[:, 0][0]
    one = severe_toxic_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        st.write('Neutral for Toxic Category')
    elif one > 0.58:
        st.write('toxic')
    else:
        st.write('Non  toxic')

    vect = threat.transform([Input])
    zero = threat_model.predict_proba(vect)[:, 0][0]
    one = threat_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        st.write('Neutral for Threat Category')
    elif one > 0.58:
        st.write('Threat')
    else:
        st.write('Non Threat')


    vect = obscene.transform([Input])
    zero = obscene_model.predict_proba(vect)[:, 0][0]
    one = obscene_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        st.write('Neutral for Obscene Category')
    elif one > 0.58:
        st.write('Obscene')
    else:
        st.write('Non Obscene')

  
    vect = insult.transform([Input])
    zero = insult_model.predict_proba(vect)[:, 0][0]
    one = insult_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        st.write('Neutral for Insult Category')
    elif one > 0.58:
        st.write('Insult')
    else:
        st.write('Non Insult')

    
    vect = identity_hate.transform([Input])
    zero = identity_hate_model.predict_proba(vect)[:, 0][0]
    one = identity_hate_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        st.write('Neutral for Identity hate Category')
    elif one > 0.58:
        st.write('Identity hate')
    else:
        st.write('Non Identity hate')


if __name__ == '__main__':
    main()
