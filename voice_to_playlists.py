import librosa as lb
import soundfile as sf
import numpy as np
import os, glob, pickle        
import sounddevice as sd
import time
import requests
import webbrowser
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.io.wavfile import write

emotion_labels = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

focused_emotion_labels = ['happy', 'sad', 'angry']

def audio_features(file_title, mfcc, chroma, mel):
    with sf.SoundFile(file_title) as audio_recording:
        audio = audio_recording.read(dtype="float32")
        sample_rate = audio_recording.samplerate

        if chroma:
            stft=np.abs(lb.stft(audio))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(lb.feature.melspectrogram(audio, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result

def get_emotion_form_file_name(file_name):
    return emotion_labels[file_name.split("-")[2]]

def loading_audio_data():
    x = [] # Input - features
    y = [] # Output - labels emotions

    # Going through all sound files
    for file in glob.glob("data//Actor_*//*.wav"):
        file_name = os.path.basename(file)
        emotion = get_emotion_form_file_name(file_name)

        # Work only with sounds that are part of focused_emotion_labels 
        if emotion in focused_emotion_labels:
            try:
                feature = audio_features(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)
            except:
                print("This file wasn't process because of an error: " + file)
                
    # Split the dateset, most for training it, and some for testing it accuracy  
    final_dataset = train_test_split(np.array(x), y, test_size=0.1, random_state=9)
    return final_dataset

def record_sound():
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    print("Recoreding in 3")
    time.sleep(1) 
    print("Recoreding in 2")
    time.sleep(1) 
    print("Recoreding in 1")
    time.sleep(1) 
    
    # Record and save
    my_recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("Recoreding: Started")
    sd.wait()
    print("Recoreding: Stopped")
    write('output.wav', fs, my_recording)

    return glob.glob("output.wav")[0]

def get_playlist(mood):
    # Sign up to rapidAPI, subscribe to this end point, and obtain your headers (x-rapidapi-key)
    # https://rapidapi.com/shekhar1000.sc/api/unsa-unofficial-spotify-api/

    url = "https://unsa-unofficial-spotify-api.p.rapidapi.com/search"

    querystring = {"query": mood,"count":"10","type":"playlists"}

    headers = {
        'x-rapidapi-key': "06b6013060msh678afa5c6a5cf22p116a90jsn8b2b444ad800",
        'x-rapidapi-host': "unsa-unofficial-spotify-api.p.rapidapi.com"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    playlist_id = response.json()["Results"][random.randint(0,9)]["id"]

    return playlist_id

def open_playlist_in_browser(playlist_id):
    webbrowser.open('https://open.spotify.com/playlist/' + str(playlist_id))

def train_model():
    # Prep
    X_train, X_test, y_train, y_test = loading_audio_data()

    # Create and train modal
    model = MLPClassifier(hidden_layer_sizes = (200,), learning_rate = "adaptive", max_iter = 400)
    model.fit(X_train,y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Model Prediction Accuracy Score
    accuracy = accuracy_score(y_true = y_test, y_pred = y_pred) * 100
    print ("Accuracy of Recognizer is: %.2f" % accuracy)

    return model, accuracy

def recognize_your_mood(model):
    while True:
        my_sound_file = record_sound()

        feature = audio_features(my_sound_file, mfcc=True, chroma=True, mel=True)
        mood_prediction = model.predict([feature])[0]

        print("Are you " + mood_prediction + "? type yes/no")
        if (input() == "yes"):
            return mood_prediction

def main():
    # Training modal (TODO: should probably save the result and not run every time)
    model, accuracy = train_model()

    if accuracy > 75:
        mood = recognize_your_mood(model)
        playlist_id = get_playlist(mood)
        open_playlist_in_browser(playlist_id)

if __name__ == "__main__":
    main()

# ----- Some Info -----
# Dataset: RAVDESS - 60 audio clips of each actor X 24 actors = 1440 audio clips