import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# Function to generate more data
def generate_more_data(base_df, num_entries):
    courses = base_df['Course'].unique()
    days_of_week = base_df['DayOfWeek'].unique()
    time_slots = base_df['TimeSlot'].unique()

    new_data = []

    for _ in range(num_entries):
        course = np.random.choice(courses)
        day_of_week = np.random.choice(days_of_week)
        time_slot = np.random.choice(time_slots)
        base_attendance = base_df[(base_df['Course'] == course) & (base_df['DayOfWeek'] == day_of_week) & (base_df['TimeSlot'] == time_slot)]['AttendanceRate']

        if not base_attendance.empty:
            attendance_rate = base_attendance.values[0] + np.random.normal(0, 2)  # small variation
        else:
            attendance_rate = np.random.uniform(70, 100)  # random if no base value

        new_data.append([course, day_of_week, time_slot, attendance_rate])

    return pd.DataFrame(new_data, columns=['Course', 'DayOfWeek', 'TimeSlot', 'AttendanceRate'])

# Load the dataset
df = pd.read_csv('attendance_data.csv')

# Generate more data
additional_data = generate_more_data(df, 10000)  # Generate 10000 additional entries

# Combine with the base dataset
df_expanded = pd.concat([df, additional_data], ignore_index=True)

# Perform one-hot encoding on categorical columns
df_expanded = pd.get_dummies(df_expanded, columns=['Course', 'DayOfWeek', 'TimeSlot'])

# Split data into features (X) and target (y)
X = df_expanded.drop('AttendanceRate', axis=1)
y = df_expanded['AttendanceRate']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train multiple machine learning models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)

# Create a Streamlit app
st.title('Attendance Prediction App')

# Training Results
st.subheader('Training Results')
for name, model in models.items():
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    st.write(f"**{name} - Training MAE:** {mae_train:.2f}, **Training R2:** {r2_train:.2f}")

# Testing Results
st.subheader('Testing Results')
for name, model in models.items():
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    st.write(f"**{name} - Testing MAE:** {mae_test:.2f}, **Testing R2:** {r2_test:.2f}")

# Predicting attendance with all models
st.sidebar.title('Predict Attendance')

course = st.sidebar.selectbox('Select Course', df['Course'].unique())
day_of_week = st.sidebar.selectbox('Select Day of Week', df['DayOfWeek'].unique())
time_slot = st.sidebar.selectbox('Select Time Slot', df['TimeSlot'].unique())

if st.sidebar.button('Predict'):
    input_data = pd.DataFrame({
        'Course': [course],
        'DayOfWeek': [day_of_week],
        'TimeSlot': [time_slot]
    })
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_data)
        predictions[name] = pred[0]

    st.subheader('Predictions')
    for name, prediction in predictions.items():
        st.write(f"{name} prediction: {prediction:.2f}")
