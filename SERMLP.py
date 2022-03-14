#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa soundfile numpy sklearn pyaudio')


# In[2]:


pip install librosa


# In[3]:


pip install pydub


# In[4]:


import librosa
import soundfile
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from IPython.display import Audio
from sklearn.preprocessing import StandardScaler,OneHotEncoder


# In[6]:


RavdessData="C:\\Users\\Hp\\SER\\Audio_Speech_Actors_01-24\\"


# In[7]:


ravdir=os.listdir(RavdessData)
paths = []
labels = []
for dir in ravdir:
    actor=os.listdir(RavdessData+dir)
    for filename in actor:
        label = filename.split('.')[0]
        label = label.split('-')
        labels.append(int(label[2]))
        paths.append(RavdessData+dir+'/'+filename)
paths=pd.DataFrame(paths,columns=['Path'])
labels=pd.DataFrame(labels,columns=['Emotions'])

        


# In[8]:


Ravdess_df=pd.concat([labels,paths],axis=1)


# In[9]:


Ravdess_df.head()


# In[10]:


Ravdess_df.shape


# In[11]:


Ravdess_df.Emotions.replace({1:'neutral',
  2:'calm',
  3:'happy',
  4:'sad',
  5:'angry',
  6:'fearful',
  7:'disgust',
  8:'surprised'},inplace=True)

Ravdess_df.head()


# In[12]:



def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result


# In[13]:


plt.title('Count of Emotions', size=16)
sns.countplot(Ravdess_df.Emotions)
plt.ylabel('Count',size=12)
plt.xlabel('Emotions',size=12)
sns.despine(top=True , right=True,left=False,bottom=False)
plt.show()


# In[14]:


import librosa.display
x, sr = librosa.load('C:\\Users\\Hp\\SER\\Audio_Speech_Actors_01-24\\Actor_01\\03-01-01-01-01-01-01.wav')
plt.figure(figsize=(10, 5))
librosa.display.waveplot(x, sr=sr)
plt.title('Waveplot - Neutral')
Audio(data=x, rate=sr)


# In[15]:


path=np.array(Ravdess_df.Path)[1]
data,samplerate=librosa.load(path)


# In[16]:


def noise(data):
    noiseamp=0.035*np.random.uniform()*np.amax(data)
    data=data + noiseamp*np.random.normal(size=data.shape[0])
    return data

x=noise(data)
plt.figure(figsize=(14,4))
librosa.display.waveplot(y=x,sr=samplerate)
Audio(x,rate=samplerate)


# In[17]:


def stretch(data , rate=0.8):
    return librosa.effects.time_stretch(data,rate)

x=stretch(data)
plt.figure(figsize=(14,4))
librosa.display.waveplot(y=x,sr=samplerate)
Audio(x,rate=samplerate)


# In[18]:



def pitch(data,samplingrate,pitchfactor=0.7):
    return librosa.effects.pitch_shift(data,samplingrate,pitchfactor)

x=pitch(data,samplerate)
plt.figure(figsize=(14,4))
librosa.display.waveplot(y=x,sr=samplerate)
Audio(x,rate=samplerate)


# In[19]:


y, sr = librosa.load('C:\\Users\\Hp\\SER\\Audio_Speech_Actors_01-24\\Actor_01\\03-01-03-02-02-01-01.wav')
librosa.feature.melspectrogram(y=y, sr=sr)


# In[20]:


D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)


# In[21]:


S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)


# In[22]:


plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram ')
plt.tight_layout()


# In[23]:


y, sr = librosa.load('C:\\Users\\Hp\\SER\\Audio_Speech_Actors_01-24\\Actor_01\\03-01-03-02-02-01-01.wav')
librosa.feature.chroma_stft(y=y, sr=sr)


# In[24]:


D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)


# In[25]:


S = np.abs(librosa.stft(y, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
chroma


# In[26]:



plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()


# In[27]:



emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'fearful', 'disgust']


# In[28]:


def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\Hp\\SER\\Audio_Speech_Actors_01-24\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        #converting stereo audio to mono
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export(file, format="wav")
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    for file in glob.glob("C:\\Users\\Hp\\SER\\Audio_Song_Actors_01-24\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        #converting stereo audio to mono
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export(file, format="wav")
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# In[29]:


#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)


# In[30]:


print((x_train.shape[0], x_test.shape[0]))


# In[31]:


print(f'Features extracted: {x_train.shape[1]}')


# In[32]:


#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[33]:


#Train the model
model.fit(x_train,y_train)


# In[34]:


#Predict for the test set
y_pred=model.predict(x_test)


# In[35]:


#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))


# In[36]:


x_axis = range(0, model.n_iter_)
fig, ax = plt.subplots()
ax.plot(x_axis, model.loss_curve_, label='Train')
ax.legend()
plt.ylabel('Classification Error')
plt.title('MLP Classification Error')
plt.show()


# In[37]:


from sklearn.metrics import classification_report
print("mlp: {}".format(classification_report(y_test, y_pred)))


# In[38]:


pickling_on = open("model.pickle","wb")
pickle.dump([model,accuracy], pickling_on)
pickling_on.close()


# In[ ]:




