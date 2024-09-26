import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.inspection import permutation_importance
from streamlit_lottie import st_lottie
import requests
import base64
from streamlit_lottie import st_lottie_spinner
import time

spinner = 'footballani.json'



# Function to create the Keras model
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(np.unique(y)), activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Streamlit UI
st.title("Football Possession Analysis")
st.write("Upload a football dataset, train a neural network model, view feature importance, and predict match scores.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Store original team and opponent names
    original_teams = data['team'].copy()
    original_opponents = data['opponent'].copy()

    # Display the dataset
    st.subheader("Uploaded Dataset")
    st.dataframe(data)

    # Data Preprocessing
    data.fillna(0, inplace=True)  # Handle missing values
    label_encoder = LabelEncoder()
    data['team'] = label_encoder.fit_transform(data['team'])
    data['opponent'] = label_encoder.fit_transform(data['opponent'])

    # Convert boolean-like columns to numeric
    bool_columns = ['shot', 'goal', 'wentback', 'possess', 'Short', 'Medium', 'Long', 'No_passes']
    for col in bool_columns:
        data[col] = data[col].apply(lambda x: 1 if x == 'T' else 0)

    # Feature Selection
    features = ['possession_number', 'shot', 'goal', 'wentback', 'timediff', 'players_behind', 'possess', 'Short', 'Medium', 'Long', 'No_passes']
    X = data[features]
    y = data['GF']  # Assuming 'GF' (Goals For) is the target variable

    # Encode target labels to ensure they are contiguous integers
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Wrap the model using KerasClassifier
    wrapped_model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=32, verbose=0)

    # Train the wrapped model
    with st_lottie_spinner(spinner, key="download", height=100, width=100):
        wrapped_model.fit(X_train, y_train, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # Permutation Feature Importance
    perm_importance = permutation_importance(wrapped_model, X_test, y_test, scoring='accuracy', n_repeats=10, random_state=42)
    feature_importance = perm_importance.importances_mean

    # Display feature importance
    st.subheader("Feature Importance")
    for i, feature in enumerate(features):
        st.write(f'Feature: {feature}, Importance: {feature_importance[i]:.4f}')

    # Build and train the main neural network model
    model = create_model()
    with st_lottie_spinner(spinner, key="download1", height=100, width=100):
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    #st.subheader(f'Test Accuracy: {accuracy:.2f}')

    # Predict the scores for the first 10 matches
    first_10_matches = X[:10]
    first_10_matches_scaled = scaler.transform(first_10_matches)
    predictions = model.predict(first_10_matches_scaled)

    # Convert predictions back to original labels
    predicted_scores = label_encoder_y.inverse_transform(np.argmax(predictions, axis=1))

    # Create a DataFrame for displaying results
    results_df = pd.DataFrame({
        'Match': range(1, 11),
        'Team': original_teams.iloc[:10],
        'Opponent': original_opponents.iloc[:10],
        'GF': data['GF'].iloc[:10],
        'Predicted Score': predicted_scores
    })

    # Display the results as a table
    st.subheader("Predicted Scores for the First 10 Matches")
    st.table(results_df)


