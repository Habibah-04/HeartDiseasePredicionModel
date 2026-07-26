import os
import pandas as pd
import streamlit as st
from google import genai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


df = pd.read_csv("heart.csv")

x = df.drop("target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)

#model select
log_model = LogisticRegression(max_iter=1000,random_state=42)
tree_model = DecisionTreeClassifier(max_depth=6,random_state=42)
forest_model = RandomForestClassifier(n_estimators=200,random_state=42)

#data fit in model 
log_model.fit(x_train_scaler,y_train)
tree_model.fit(x_train,y_train)
forest_model.fit(x_train,y_train)

#model prediction
log_pred = log_model.predict(x_test_scaler)
tree_pred = tree_model.predict(x_test)
forest_pred = forest_model.predict(x_test)

#accuracy 
log_acc = accuracy_score(y_test,log_pred)
tree_acc = accuracy_score(y_test,tree_pred)
forest_acc = accuracy_score(y_test,forest_pred)

model_result = pd.DataFrame({
    "Model":["Logistic Regression","Decision Tree","Random Forest"],
    "Accuracy":[log_acc,tree_acc,forest_acc]
})

model_result = model_result.sort_values(by="Accuracy",ascending=False)
st.subheader("Model Comparison")
st.dataframe(model_result,use_container_width=True)

best_model_name = model_result.iloc[0]["Model"]
best_accuracy = model_result.iloc[0]["Accuracy"]

st.success(f"Best Model : {best_model_name}")
st.info(f"Accuracy : {best_accuracy*100:.2f}%")

c1,c2,c3 = st.columns(3)
with c1:
    st.metric("Logistic",f"{log_acc*100:.2f}%")
with c2:
    st.metric("Decision Tree",f"{tree_acc*100:.2f}%")
with c3:
    st.metric("Random Forest",f"{forest_acc*100:.2f}%")

st.divider()
st.subheader("Data Visualization")
tab1,tab2,tab3 = st.tabs(["Distribution","Relationship","Heatmap"])
with tab1:
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Heart Disease Distribution")
        fig,ax = plt.subplots(figsize=(4,3))
        target = df["target"].value_counts()
        ax.bar(["Healthy","Disease"],target.values)
        st.pyplot(fig)

    with c2:
        st.subheader("Gender Distribution")
        fig,ax = plt.subplots(figsize=(4,3))
        gender = df["gender"].value_counts()
        ax.bar(["Female","Male"],gender.values)
        st.pyplot(fig)

    c3,c4 = st.columns(2)
    with c3:
        st.subheader("Age Distribution")
        fig,ax = plt.subplots(figsize=(4,3))
        ax.hist(df["age"],bins=20)
        ax.set_xlabel("Age")
        st.pyplot(fig)


    with c4:
        st.subheader("Cholesterol Distribution")
        fig,ax = plt.subplots(figsize=(4,3))
        ax.hist(df["chol"],bins=20)
        ax.set_xlabel("Cholesterol")
        st.pyplot(fig)

with tab2:
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Age vs Cholesterol")
        fig,ax = plt.subplots(figsize=(4,3))
        ax.scatter(df["age"],df["chol"],alpha=0.6,c=df["age"],cmap="cool",s=10)
        ax.set_xlabel("Age")
        ax.set_ylabel("Cholesterol")
        st.pyplot(fig)

    with c2:
        st.subheader("BP vs HR")
        fig,ax = plt.subplots(figsize=(4,3))
        ax.scatter(df["trestbps"],df["thalachh"],alpha=0.6,c=df["trestbps"],cmap="ocean",s=10)
        ax.set_xlabel("Blood Pressure")
        ax.set_ylabel("Heart Rate")
        st.pyplot(fig)

with tab3:
    st.subheader("Correlation Heatmp")
    fig,ax = plt.subplots(figsize=(8,5))
    heat = ax.imshow(df.corr(),cmap="coolwarm")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns,rotation=90,fontsize=8)
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns,fontsize=8)
    plt.colorbar(heat)
    st.pyplot(fig)

st.divider()
st.header("Heart Disease Prediction")
c1,c2 = st.columns(2)
with c1:
    age = st.number_input("Age :",18,100,20)
    gender = st.selectbox("Gander :",[0,1],format_func = lambda x:"Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type :",[0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure :",80,250,120)
    chol = st.number_input("Cholesterol :",100,600,200)
    fbs = st.selectbox("Fasting Blood Sugar :",[0,1])
    restecg = st.selectbox("Resting Electrocardiographic :",[0,1,2])
    

with c2:
    thalachh = st.number_input("Maximum Heart Rate :",60,220,150)
    exang = st.selectbox("Excersise :",[0,1],format_func = lambda x:"No" if x==0 else "Yes")
    oldpeak = st.number_input("Old Peak :",0.0,10.0,1.0)
    slope = st.selectbox("Slope :",[0,1,2])
    ca = st.selectbox("Major Vessels :",[0,1,2,3,4])
    thal = st.selectbox("Thalassemia :",[0,1,2,3])



st.divider()
if st.button("Predict",use_container_width=True):
    patient = pd.DataFrame([{
        "age":age,
        "gender":gender,
        "cp":cp,
        "trestbps":trestbps,
        "chol":chol,
        "fbs":fbs,
        "restecg":restecg,
        "thalachh":thalachh,
        "exang":exang,
        "oldpeak":oldpeak,
        "slope":slope,
        "ca":ca,
        "thal":thal
    }])

    if best_model_name =="Logistic Regression":
        result = log_model.predict(scaler.transform(patient))
    elif best_model_name =="Decision Tree":
        result = tree_model.predict(patient)
    else:
        result = forest_model.predict(patient)
    
    st.subheader("Prediction Result")
    if result[0] ==1:
        st.error("High Risk of Heart Disease")
        cp_map={
            0:"Typical Angina",
            1:"Atypical Agina",
            2:"Non-Angina Pain",
            3:"Asymptomatic"
        }
        restecg_map={
            0:"Normal",
            1:"ST-T wave Abnormality",
            2:"Left Venticular Hypertrophy"
        }
        slope_map={
            0:"Upsloping",
            1:"Flat",
            2:"Downsloping"
        }
        thal_map={
            0:"Normal",
            1:"Fixed Defect",
            2:"Revercible Defect",
            3:"Unknow"
        }
        prompt=f"""
        You are a experienced cardiologist.
        A pateint's heart disease predicion model classified this patient as HIGH RISK.
        Pateint Details
        Age : {age}
        Gender :{"Male" if gender==1 else "Female"}
        Chest Pain Type :{cp_map[cp]}
        Resting Blood Pressure : {trestbps}
        Cholesterol : {chol}
        Fasting Blood Sugar :{"High" if fbs ==1 else "Normal"}
        Rest ECG :{restecg_map[restecg]}
        Maximum Heart Rate :{thalachh}
        Excercise Engina :{"Yes" if exang==1 else "No"}
        Old Peak :{oldpeak}
        Slope : {slope_map[slope]}
        Major Vessels :{ca}
        Thalassemia :{thal_map[thal]}

        Give response in this format.
        Risk Level
        Possible Reason
        Life Style Advice
        Diet Recommendation
        Excercise Recommendation
        Doctor Recommendation
        Keep response under 250 word
        """
        st.subheader("Ai Health Suggestion")
        try:
            with st.spinner("Generating AI Report....."):
                response = client.models.generate_content(model="gemini-flash-latest",contents=prompt)
            st.markdown(response.text)
        except Exception:
            st.warning("AI Server Busy. Showing Default Health Recommendation.")
            st.info("""
            - Consult With Doctor
            - Monitor Blood Pressure
            - Reduce Cholesterol
            """)


    else:
        st.success("Low Rish of Heart Diseas")
        st.subheader("AI Health Suggestion")
        st.info("""
        - Continue Healthy Lifestyle
        - Regular Exercise
        - Balanced Diet
        - Drink Enough Water
        - Sleep 8 hrs daily
        - Annual Health Checkup
        """)

