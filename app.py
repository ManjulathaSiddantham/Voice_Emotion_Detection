from flask import Flask, render_template,request, redirect, session
import pandas as pd
import numpy as np
import os
import seaborn as sns
# import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
from keras.models import load_model

warnings.filterwarnings('ignore')

app = Flask(__name__)
# Generate a secret key
import secrets
secret_key = secrets.token_hex(16)

# Set the secret key for session management
app.secret_key = secret_key

# User credentials
users = {}

@app.route('/')
def new_home():
    if 'username' in session:
        # If user is already logged in, redirect to index page
        return redirect('/index')
    else:
        # Otherwise, show the login page
        return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username in users and users[username]['password'] == password:
        # If credentials are valid, create a session for the user
        session['username'] = username
        return redirect('/index')
    else:
        # If credentials are invalid, show an error message
        error_message = 'Invalid username or password'
        return render_template('login.html', error_message=error_message)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']


        if username in users:
            # If username already exists, show an error message
            error_message = 'Username already exists. Please choose a different username.'
            return render_template('register.html', error_message=error_message)
        else:
            # Register the user
            users[username] ={'password': password, 'email': email}
            # Create a session for the user
            session['username'] = username
            return redirect('/index')
    else:
        return render_template('register.html')

@app.route('/index')
def index():
    if 'username' in session:
        # If user is logged in, show the index page
        return render_template('index.html', username=session['username'])
    else:
        # If user is not logged in, redirect to the login page
        return redirect('/')

@app.route('/logout')
def logout():
    # Clear the session and redirect to the login page
    session.clear()
    return redirect('/')

@app.route('/about',methods=['POST','GET'])
def about():
    return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Process the contact form data (e.g., send an email, save to database, etc.)
        # You can add your own logic here
        
        return render_template('contact.html', message_sent=True)
    else:
        return render_template('contact.html', message_sent=False)


upload_folder = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
# upload_folder = 'uploads'
# app.config['UPLOAD_FOLDER'] = upload_folder

model = load_model('lstm_model.h5')  # Load your trained model here

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['audio']
    # audio_path = 'uploaded_audio.wav'
    # audio_path = 'C:/Users/md221/OneDrive/Desktop/mini_webpart/mini-2/Upload-folder'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_audio.wav')
    print(f"Audio Path: {audio_path}")
    audio_file.save(audio_path)

    # Load the LSTM model
    model = load_model('lstm_model.h5')

    # Function to extract MFCC features
    def extract_mfcc(filename):
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    # Function to make prediction
    def predict_emotion(filename):
        # Extract MFCC features
        mfcc = extract_mfcc(filename)

        # Reshape the MFCCs to match the input shape of the LSTM model
        reshaped_mfcc = np.reshape(mfcc, (1, mfcc.shape[0], 1))

        # Make the prediction using the loaded model
        prediction = model.predict(reshaped_mfcc)

        # Get the index of the maximum value in the predictions array
        max_index = np.argmax(prediction)

        # Map the emotion label to the corresponding emotion
        emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        predicted_emotion = emotion_labels[max_index]

        return predicted_emotion

    # Example usage
    filename = audio_path
    predicted_emotion = predict_emotion(filename)

    return render_template('index.html', predicted_emotion=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)
