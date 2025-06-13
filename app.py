import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import pyttsx3

def voice_out(havetosay):
    initiate = pyttsx3.init()
    initiate.setProperty('rate', 190) 
    voices = initiate.getProperty('voices')
    initiate.setProperty('voice', voices[0].id) 
    initiate.say(havetosay)
    initiate.runAndWait()

st.set_page_config(layout='wide', page_title='Credit Card Fraud Detection', page_icon='💳')

data = pd.read_csv("C:/Users/Gopi/Desktop/Credit-Card-Fraudulant-Transaction-Detection-Model-main/creditcard.csv")

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = RandomForestClassifier() 
model.fit(X_train, y_train)

train_acc = accuracy_score(model.predict(X_train), y_train)*100
test_acc = accuracy_score(model.predict(X_test), y_test)*100

st.title("Credit Card Fraud Detection Model")

import random as rd
ran = rd.randint(0, 983)

col3, col4, col5 = st.columns([6,0.5,3.5])
with col5:
    image_path = 'C:/Users/Gopi/Desktop/Credit-Card-Fraudulant-Transaction-Detection-Model-main/image1.png'
    img = Image.open(image_path)
    st.image(img)
with col3:
    col1, col2 = st.columns([8,2])
    with col1:
        st.subheader("Generate Random Values")
    with col2:
        ''
        random = st.button('Random')
    if random:
        random_row = X.iloc[ran,:]
        crandom_row = ', '.join(map(str, random_row))
        st.code(crandom_row)
    ''

    # st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")
    input_df = st.text_input('Enter All the Features')
    col6, col7 = st.columns([8,2])
    with col7:
        input_df_lst = input_df.split(',')
        submit = st.button("Submit")
    if submit:  
        features = np.array(input_df_lst, dtype=np.float64)
        prediction = model.predict(features.reshape(1,-1))

        if prediction[0] == 0:
            st.write("The Input Features represent the **LEGITIMATE** Transaction")
            havetosay = "The Input Features represent the LEGITIMATE Transaction"
            voice_out(havetosay)
        else:
            st.write("The Input Features represent the **FRAUDULENT** Transaction")
            havetosay = "The Input Features represent the FRAUDULENT Transaction"
            voice_out(havetosay)
            
        
