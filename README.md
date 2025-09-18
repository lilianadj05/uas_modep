# 🫃🏻Obesity Prediction Model

## 📌 Project Overview
This project aims to predict an individual's obesity level based on various factors such as lifestyle, eating habits, and physical activity. The solution consists of a machine learning model deployed as a web application using FastAPI and Streamlit, allowing users to input their data and receive a prediction in real time.

## 📊 Dataset
The dataset used in this project contains data on individuals' gender, age, height, weight, family history of overweight, eating habits, physical activity, and technology usage. The target variable is the individual's obesity level, categorized into seven classes ranging from "Insufficient Weight" to "Obesity Type III."

## 🔍 Key Findings
1. The dataset contained 10 duplicated rows and anomalies in the "Age" column, which were addressed during the data cleaning process.
2. The target variable, obesity level, was found to be well-balanced across all seven categories.
3. Several numeric variables like "Age," "Weight," and "NCP" had outliers, but they were kept to preserve valuable information.

## 📈 Steps
1. **Data Cleaning**: Duplicated rows were removed, and the "Age" column was cleaned by converting string values (e.g., '21 years') to a numeric format.
2. **Preprocessing**: A preprocessing pipeline was created to handle missing values using a median imputer for numeric data and a most-frequent imputer for categorical data. Categorical features were encoded using OrdinalEncoder.
3. **Modeling and Evaluation**: The data was split into training and testing sets. Two models, **Random Forest** and **XGBoost**, were trained and evaluated. The XGBoost model was selected as the best performer, achieving a **93% accuracy score**. The recall score was prioritized to minimize cases of obesity going undetected.
4. **Deployment**: The XGBoost model and the label encoder were saved as pickle files. These files were loaded into a **FastAPI** backend to create a prediction API. A **Streamlit** application was built as the frontend, allowing users to interact with the model by sending data to the FastAPI endpoint to receive a prediction.

## 💡 Conclusion
The project successfully developed and deployed an XGBoost model that can accurately predict a person's obesity level with a high accuracy of 93%. This tool can serve as an accessible and user-friendly way to assess one's obesity risk based on key health and lifestyle indicators.

## 🔮 Possible Future Works
1. Integrate the Body Mass Index (BMI) calculation into the prediction to provide more context and a common health metric.
2. Deploy the application on a cloud platform (e.g., Heroku, AWS, or Google Cloud) to make it publicly accessible.
3. Enhance the frontend with a feature that provides personalized recommendations for diet and exercise based on the prediction result.

## 👨‍💻 Author
**Liliana Djaja Witama** | Undergraduate Data Science Student at BINUS University
