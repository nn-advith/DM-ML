import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(
    page_title="Heart Disease",
    page_icon="heart",
)

st.write("""
# Heart Disease Classification
***
""")



##########---------------------------------------------------------------------------------------------------------------------------------------------
#Sidebar

st.sidebar.write("""***""")
disdata = st.sidebar.checkbox("Display Dataset")

st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ( "Logistic Regression", "Random Forest"))

st.sidebar.write("""***""")
st.sidebar.header("Input Parameters")
bmi = st.sidebar.number_input('Body Mass Index')
smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]", ["Yes", "No"])
alcohol = st.sidebar.selectbox("Do you consume more than 14 drinks(men) or 7 drinks(women) per week?", ["Yes", "No"])
stroke = st.sidebar.selectbox("Have you ever had a stroke?", ["Yes", "No"])
ph = st.sidebar.number_input('How many days in the past 30 days has your physical health not been good?', min_value=0, max_value=30)
mh = st.sidebar.number_input('How many days in the past 30 days has your mental health not been good?', min_value=0, max_value=30)
dw = st.sidebar.selectbox("Do you have serious difficulty walking or climbing stairs?", ["Yes", "No"])
sex = st.sidebar.selectbox("Are you male or female?", ["Male", "Female"])
ac = st.sidebar.selectbox("Age-Category", ["18-24", "25-29", "30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","74-79","80 or older"])
race = st.sidebar.selectbox("Race", ["American Indian/ Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"])
diabetic = st.sidebar.selectbox("Are you diabetic?", ["Yes", "Yes (during pregnancy)", "No", "No (Borderline Diabetic)"])
pa = st.sidebar.selectbox("Have you done any physical activity in the last 30 ays other than regular work?", ["Yes", "No"])
genH = st.sidebar.selectbox("Describe your general health:", ["Excellent", "Fair", "Good", "Poor", "Very Good"])
stime = st.sidebar.number_input('On average, how many hours of sleep do you get in a 24-hour period?')
asthma = st.sidebar.selectbox("Are you asthmatic?", ["Yes", "No"])
kd = st.sidebar.selectbox("Do you have any kidney disease?", ["Yes", "No"])
sc = st.sidebar.selectbox("Do you have Skin Cancer?", ["Yes", "No"])

st.sidebar.write("""***""")
pr = st.sidebar.button("Predict")

##########---------------------------------------------------------------------------------------------------------------------------------------------

yesno_dict = {"Yes": [0,1], "No": [1,0]}
sex_dict = {"Male": [0,1], "Female": [1,0]}
age_dict = ["18-24", "25-29", "30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","74-79","80 or older"]
age_list = [0]*13 
race_dict = ["American Indian/ Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"]
race_list = [0]*6
diabetic_dict = ["Yes", "Yes (during pregnancy)", "No", "No (Borderline Diabetic)"]
diabetic_list = [0, 0, 0, 0]
gh_dict = ["Excellent", "Fair", "Good", "Poor", "Very Good"]
gh_list = [0]*5


def findOne(arr):
    for i in range(len(arr)):
        if arr[i] == 1:
            return 1
    
    return -1


@st.cache(persist= True)
def load():
    data= pd.read_csv("./dataset/h1.csv")
    return data

df = load()

st.subheader("Dataset")
st.write("Personal Key Indicators of Heart Disease -2020 annual CDC survey data of 400k adults related to their health status")
if disdata:
    st.write(df)
else:
    st.write("Enable 'Display Dataset' to show dataset")

st.write("""***""")

def getIP():

    age_list[age_dict.index(ac)] = 1
    race_list[race_dict.index(race)] = 1
    diabetic_list[diabetic_dict.index(diabetic)] = 1
    gh_list[gh_dict.index(genH)] = 1
    ip = yesno_dict[smoking]+ yesno_dict[alcohol]+yesno_dict[stroke]+yesno_dict[dw]+ sex_dict[sex]+age_list+race_list+diabetic_list+yesno_dict[pa]+gh_list+yesno_dict[asthma]+yesno_dict[kd]+yesno_dict[sc]+[bmi, ph,mh,stime]
    
    return ip
    


if pr:
    st.session_state.ip = getIP();
    filename = ('LR.sav' if classifier == 'Logistic Regression' else 'RFC.sav') 
    loaded_model = pickle.load(open(filename, 'rb'))

    if 'res' not in st.session_state:
        st.session_state.res = loaded_model.predict([st.session_state.ip])
    else:
        st.session_state.res = loaded_model.predict([st.session_state.ip])


if 'res' not in st.session_state:
    st.subheader("Please set input parameters and \"Predict\"")
else:
    userip = {
        "BMI": str(st.session_state.ip[-4]),
        "Smoking": list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[0:2])],
        "Alcohol": list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[2:4])],
        "Stroke" : list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[4:6])],
        "PhysicalHealth": str(st.session_state.ip[-3]),
        "MentalHealth": str(st.session_state.ip[-2]),
        "DiffWalking": list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[6:8])],
        "Sex": list(sex_dict.keys())[list(sex_dict.values()).index(st.session_state.ip[8:10])],
        "AgeCategory": age_dict[findOne(st.session_state.ip[10:23])],
        "Race": race_dict[findOne(st.session_state.ip[23:29])],
        "Diabetic": diabetic_dict[findOne(st.session_state.ip[29:33])],
        "PhysicalActivity": list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[33:35])],
        "GeneralHealth":  gh_dict[findOne(st.session_state.ip[35:40])],
        "SleepTime": str(st.session_state.ip[-1]),
        "Asthma": list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[40:42])],
        "KidneyDisease": list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[42:44])],
        "SkinCancer": list(yesno_dict.keys())[list(yesno_dict.values()).index(st.session_state.ip[44:46])]
    }

    
    features = pd.DataFrame.from_dict(userip, orient='index')
    features = features.rename({0: 'Values'}, axis="columns")

    st.subheader("Results:")
    st.write("""#####  Input Parameters: """)
    st.write(features)
    st.write("""#####  Selected Classifier: """)
    st.write(classifier)
    st.write("""#####  Possible Heart Disease?""")
    st.write("Yes" if st.session_state.res == 1 else "No")  