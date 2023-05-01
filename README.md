# Importing libraries
import librosa
import parselmouth
from parselmouth.praat import call
import csv
import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading the dataset
df = pd.read_csv("parkinsons.csv")
df.head()

# Splitting the features and the target
X = df.drop(["name", "status"], axis=1)
y = df["status"]

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the SVM model
svm = SVC(kernel="rbf", gamma="auto")
svm.fit(X_train, y_train)

# Making predictions on the test set
y_pred = svm.predict(X_test)

# Evaluating the model performance
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", acc)
print("Confusion matrix: \n", cm)

# Saving the model
pickle.dump(svm, open("svm_model.pkl", "wb"))

# Loading the model
svm = pickle.load(open("svm_model.pkl", "rb"))

# Calculating the voice measures for a new voice recording (using the same code as before)

# Setting parameters
samplerate = 44100 # Sample rate
duration = 10 # Duration in seconds
filename = "voice.wav" # File name

# Recording audio
print("Start recording...")
data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
sd.wait() # Wait until recording is finished
print("Stop recording...")

# Saving audio
sf.write(filename, data, samplerate)

# Loading voice sample
y, sr = librosa.load(filename)
y_trimmed, index = librosa.effects.trim(y) # trim the silent parts
sf.write("voice.wav", y_trimmed, sr) # save the trimmed recording
# Calculating MDVP:Fo (Hz) - Average vocal fundamental frequency
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_mean = np.nanmean(f0)

# Calculating MDVP:Fhi (Hz) - Maximum vocal fundamental frequency
f0_max = np.nanmax(f0)

# Calculating MDVP:Flo (Hz) - Minimum vocal fundamental frequency
f0_min = np.nanmin(f0)

# Calculating MDVP:Jitter (%),MDVP:Jitter (Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency
sound = parselmouth.Sound("voice.wav")
pitch = sound.to_pitch()
pointProcess = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
jitter_percent = parselmouth.praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
jitter_abs = parselmouth.praat.call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
jitter_rap = parselmouth.praat.call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
jitter_ppq = parselmouth.praat.call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
jitter_ddp = parselmouth.praat.call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

# Calculating MDVP:Shimmer,MDVP:Shimmer (dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
shimmer_local = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local)...", 0.0 ,0.0 ,75 ,500 ,1.3 ,1.6) 
shimmer_local_db = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local_dB)", 0.0 ,0.0 ,75 ,500 ,1.3 ,1.6) 
shimmer_apq3 = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq3)...", 0.0 ,0.0 ,75 ,500 ,1.3 ,1.6) 
shimmer_apq5 = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq5)...", 0.0 ,0.0 ,75 ,500 ,1.3 ,1.6) 
shimmer_apq11 = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq11)", 0.0 ,0.0 ,75 ,500 ,1.3 ,1.6)
shimmer_dda = parselmouth.praat.call([sound, pointProcess], "Get shimmer (dda)...", 0.0 ,0.0 ,75 ,500 ,1.3 ,1.6)
# Replacing the NaN values with the mean of the column
shimmer_measures = np.array([shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda])
shimmer_measures = np.where(np.isnan(shimmer_measures), np.nanmean(shimmer_measures), shimmer_measures)

# Assigning the corrected values to the variables
shimmer_local = shimmer_measures[0]
shimmer_local_db = shimmer_measures[1]
shimmer_apq3 = shimmer_measures[2]
shimmer_apq5 = shimmer_measures[3]
shimmer_apq11 = shimmer_measures[4]
shimmer_dda = shimmer_measures[5]
print(sound)
print(pointProcess.get_number_of_points())
print(shimmer_local)
print(shimmer_local_db)
print(shimmer_apq3)
print(shimmer_apq5)
print(shimmer_apq11)
print(shimmer_dda)
# Calculating NHR,HNR - Two measures of ratio of noise to tonal components in the voice
harmonicity = parselmouth.praat.call(sound,"To Harmonicity (cc)", 0.01 ,75 ,0.1 ,4.5 )
nhr = parselmouth.praat.call(harmonicity,"Get mean",1,sound.xmax)
hnr = parselmouth.praat.call(sound,"To Harmonicity (ac)", 0.01 ,75 ,-25 ,4.5)
hnr_mean = parselmouth.praat.call(hnr,"Get mean",1,sound.xmax)
# Calculating RPDE and DFA
rpde = call(sound,"To Sound (hum)", 100 ,0.4 ,0.1 ,1 ,1 ,1 ,1 ,1 )
rpde = call(rpde,"To Sound (remove noise)", 0.025 ,80 ,10000 ,40 ,600 )
rpde = call(rpde,"To Matrix")
rpde = call(rpde,"To RecurrencePlot (determinism)", 2 ,0.00001 ,1.3 )
rpde = call(rpde,"Get recurrence period density entropy")
dfa = call(sound,"To Sound (hum)", 100 ,0.4 ,0.1 ,1 ,1 ,1 ,1 ,1 )
dfa = call(dfa,"To Sound (remove noise)", 0.025 ,80 ,10000 ,40 ,600 )
dfa = call(dfa,"To Matrix")
dfa = call(dfa,"To Polygon")
dfa = call(dfa,"ScaleX", 0.00001)
dfa = call(dfa,"To Spectrum (slice)", 1)
dfa = call(dfa,"Get lowess smoothing", 5)
# Calculating PPE
ppe = call(sound,"To Sound (hum)", 100 ,0.4 ,0.1 ,1 ,1 ,1 ,1 ,1 )
ppe = call(ppe,"To Sound (remove noise)", 0.025 ,80 ,10000 ,40 ,600 )
ppe = call(ppe,"To Pitch (cc)", 0.0, 75, 600)
ppe = call(ppe,"To PointProcess")
ppe = call(ppe,"To Sound (point)", 0.9)
ppe = call(ppe,"To Sound (change gender)", 100, 1.2, 1)
ppe = call(ppe,"To Matrix")
ppe = call(ppe,"To Polygon")
ppe = call(ppe,"ScaleX", 0.00001)
ppe = call(ppe,"To Spectrum (slice)", 1)
ppe = call(ppe,"Get lowess smoothing", 5)
# Calculating D2
d2 = call(sound,"To Sound (hum)", 100 ,0.4 ,0.1 ,1 ,1 ,1 ,1 ,1 )
d2 = call(d2,"To Sound (remove noise)", 0.025 ,80 ,10000 ,40 ,600 )
d2 = call(d2,"To Matrix")
d2 = call(d2,"To RecurrencePlot (determinism)", 2 ,0.00001 ,1.3 )
d2 = call(d2,"Get correlation dimension")
# Calculating spread1 and spread2
spread1 = call(sound,"To Sound (hum)", 100 ,0.4 ,0.1 ,1 ,1 ,1 ,1 ,1 )
spread1 = call(spread1,"To Sound (remove noise)", 0.025 ,80 ,10000 ,40 ,600 )
spread1 = call(spread1,"To Matrix")
spread1 = call(spread1,"Get standard deviation")
spread2 = call(sound,"To Sound (hum)", 100 ,0.4 ,0.1 ,1 ,1 ,1 ,1 ,1 )
spread2 = call(spread2,"To Sound (remove noise)", 0.025 ,80 ,10000 ,40 ,600 )
spread2 = call(spread2,"To Matrix")
spread2 = call(spread2,"Get skewness")
# Writing the values to a csv file
header = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR"]
data = [f0_mean, f0_max, f0_min, jitter_percent, jitter_abs, jitter_rap, jitter_ppq, jitter_ddp, shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda, nhr, hnr_mean]

with open("voice_parameters.csv", "w", encoding="UTF8", newline="") as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    # write the data
    writer.writerow(data)

# Saving the voice measures as a numpy array or a pandas dataframe
voice_measures = np.array([f0_mean, f0_max, f0_min, jitter_percent, jitter_abs, jitter_rap, jitter_ppq, jitter_ddp, shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda, nhr, hnr_mean])
# or
voice_measures = pd.DataFrame([f0_mean, f0_max, f0_min, jitter_percent, jitter_abs, jitter_rap, jitter_ppq, jitter_ddp, shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda, nhr, hnr_mean])
# Checking for infinity or very large values
np.isinf(voice_measures) # check for infinity values
np.isfinite(voice_measures) # check for finite values
np.max(voice_measures) # check for the maximum value
voice_measures = voice_measures.dropna()
# Using the model.predict method on the voice measures to get the predicted status
status = svm.predict(voice_measures)
print("Status: ", status) # 0 for healthy, 1 for PD
