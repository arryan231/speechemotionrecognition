from distutils.log import debug
from flask import Flask,render_template,request,redirect,url_for,flash
import pickle
import os
from werkzeug.utils import secure_filename
import librosa
import numpy as np


UPLOAD_FOLDER = 'C:\\Users\\Hp\\SER\\Audio_Speech_Actors_01-24\\Actor_*\\*.wav'
ALLOWED_EXTENSIONS = {'wav','mp3'}

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model=pickle.load(open('model.pickle','rb'))

def extract_feature(file_name, mfcc, chroma, mel):

    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result



@app.route('/',methods=['GET', 'POST'])

def upload():
    if (request.method == 'POST'):
        f=request.files['file1']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        #return "uploaded"
        path=os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))

        feature=np.array([extract_feature(path, mfcc=True, chroma=True, mel=True)])
        #feature= extract_feature(path, mfcc=True, chroma=True, mel=True)
        #X.append(feature)
        y_pred = model.predict(feature)
    return render_template('index.html')


@app.route("/predict",methods=['POST','GET'])
def results():
    """
    This route is used to save the file, convert the audio to 16000hz monochannel,
    and predict the emotion using the saved binary model
    """
    f = request.files['file']

    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))


    wav_file_pre  = os.listdir("./audio")[0]
    wav_file_pre = f"{os.getcwd()}/audio/{wav_file_pre}"

    model = pickle.load(open(f"{os.getcwd()}/model.model", "rb"))
    x_test =extract_feature(wav_file_pre, mfcc=True, chroma=True, mel=True)
    y_pred=model.predict(np.array([x_test]))
    os.remove(wav_file_pre)
    return render_template('predict.html', value=y_pred[0])









if __name__ == "__main__":
    app.run(debug=True)