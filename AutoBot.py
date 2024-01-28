# Developed By Siddharth G Singh -

import streamlit as st
import os
import pandas as pd

import atexit

# Profiling Libraries
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

#Auto Machine Learning Library
import pycaret.regression as regression
# import setup, compare_models, pull, save_model 
import pycaret.classification as classification


with st.sidebar:
    st.image("asset\AutoBot.png")
    st.title("AutoBot")
    choice = st.radio(
        "**Navigation** - Choose Operation",
        ["Upload Data", "Profiling", "ML-Modelling", "Download Data"],
    )
    st.info(
        """AutoBot: Your Partner in Automated Machine Learning

Experience the Power of Automation for Initial Dataset Insights

Welcome to AutoBot - your advanced assistant in the realm of automated machine learning. With AutoBot, dive into the world of data analytics with unparalleled ease and efficiency. Whether you're a seasoned data scientist or just starting out, AutoBot is designed to provide comprehensive initial insights into any dataset you wish to explore.
            
Technology Using : Streamlit, Ydata Profiling , PyCaret            """
    )

# st.write("Sid Playground")
global df 
df = None
global data 
data = None
if os.path.exists("uploaded_data.csv"):
    df = pd.read_csv("uploaded_data.csv", index_col=None)


if choice == "Upload Data":
    st.title("Upload Your Data For Analysis!")
    data = st.file_uploader("Upload Your Dataset Here")
    if data:
        df = pd.read_csv(data, index_col=None)
        df.to_csv("uploaded_data.csv", index=None)
        st.dataframe(df)


if choice == "Profiling":
    if df is not None:
        st.title("Automated Magical Data Analysis")
        pr = ProfileReport(df)
        st_profile_report(pr)
    else:
        st.warning("Dataset Is Not Available!")


if choice == "ML-Modelling":
    if df is not None:
        ML_model_df = pd.DataFrame({'Regression':1,'Classification':2}, index=[0])
        st.title("Auto Machine Learning Prediction")
        select_column = st.selectbox("Select Your Target",df.columns)
        select_ML_model = st.selectbox("Select Your Model",ML_model_df.columns)
        col1, col2 = st.columns([0.5,1])
        button_clicked = False
        with col1:
            pass
        with col2:
            if st.button('Run Auto-Modelling'):
                 button_clicked = True
        if button_clicked == True:
            if select_ML_model == 'Regression':
                regression.setup(df,target=select_column)
                setup_df = regression.pull()
                st.info("Auto-ML Settings and Tweaks.")
                st.dataframe(setup_df)
                loading_message = st.empty()  # Create an empty widget
                loading_message.text("Loading comparison data... Please wait.")
                best_model = regression.compare_models()
                compare_df = regression.pull()
                # Update the message
                loading_message.text("Data loaded successfully!")
                st.info("Performance Of Different ML Models :")
                st.dataframe(compare_df)
                st.write("Best Model(Result) : ")
                st.info(compare_df.iloc[0, 0])
                regression.save_model(best_model, 'regression_trained_model')
                st.write("Model saved successfully ! ")
            if select_ML_model == 'Classification':
                classification.setup(df,target=select_column)
                setup_df = classification.pull()
                st.info("Auto-ML Settings and ()")
                st.dataframe(setup_df)
                loading_message = st.empty()  # Create an empty widget
                loading_message.text("Loading comparison data... Please wait.")
                best_model = classification.compare_models()
                compare_df = classification.pull()
                # Update the message
                loading_message.text("Data loaded successfully!")
                st.info("Performance Of Different ML Models :")
                st.dataframe(compare_df)
                st.write("Best Model(Result) : ")
                st.info(compare_df.iloc[0, 0])
                classification.save_model(best_model, 'classification_trained_model')
                st.write("Model saved successfully ! ")

    else:
        st.warning("**No Data To Train On 😴**")

if choice == "Download Data":
    if os.path.exists("regression_trained_model.pkl"):
        with open("regression_trained_model.pkl",'rb') as f:
            st.download_button("Download Regression Model",f,file_name="regression_trained_model.pkl")
            print(f"regression_trained_model.pkl Downloaded")
    if os.path.exists("classification_trained_model.pkl"):
        with open("classification_trained_model.pkl",'rb') as f:
            st.download_button("Download Classification Model",f,file_name="classification_trained_model.pkl")
            print(f"classification_trained_model.pkl Downloaded")
    else:
        st.warning("**No Trained Models To Download 😅**")

    


#
file_path = "uploaded_data.csv"
def delete_file_andDataFrame(file_path,df=None):

    if os.path.exists("regression_trained_model.pkl"):
        os.remove("regression_trained_model.pkl")
        print(f"regression_trained_model.pkl - Removed.")
    if os.path.exists("classification_trained_model.pkl"):
        os.remove("classification_trained_model.pkl")
        print(f"classification_trained_model.pkl - Removed.")

    if os.path.exists(file_path):
        df=None
        os.remove(file_path)
        print(f"File and Data {file_path} has been deleted.")
    else:

        print(f"File {file_path} does not exist.")

    if df is not None:
        df=None
        print(f"Data Cache Has Been Cleared.")

# Register the cleanup function
atexit.register(delete_file_andDataFrame,file_path,df)