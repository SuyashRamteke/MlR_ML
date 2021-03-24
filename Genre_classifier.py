import numpy as np
import glob
import librosa
import math
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.linear_model import LogisticRegression as lr
import matplotlib.pyplot as plt

fnames = glob.glob("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/*/*.wav")


genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'rock', 'blues', 'reggae', 'pop', 'metal']

# compute the features
def mfcc(fnames):

    audio_features = np.zeros((len(fnames), 40))
    target = np.zeros(len(fnames))

    for (i, fname) in enumerate(fnames):
        print("Processing %d %s" % (i, fname))
        for (label, genre) in enumerate(genres):
            if genre in fname:
                audio, srate = librosa.load(fname)

                feature_matrix = librosa.feature.mfcc(y=audio, sr=srate)

                mean_feature = np.mean(feature_matrix,axis=1)
                print(mean_feature.shape)
                std_feature = np.std(feature_matrix, axis=1)
                audio_fvec = np.hstack([mean_feature, std_feature])
                print(audio_fvec.shape)
                audio_features[i] = audio_fvec
                target[i] = label

    return (audio_features, target)

def constant_q(audio_features, target, fnames):

    audio_features = np.zeros((len(fnames), 336))
    target = np.zeros(len(fnames))

    for (i, fname) in enumerate(fnames):
        print("Processing %d %s" % (i, fname))
        for (label, genre) in enumerate(genres):
            if genre in fname:
                audio, srate = librosa.load(fname)

                feature_matrix = np.abs(librosa.cqt(y=audio, sr=srate, n_bins=168, res_type='fft'))

                mean_feature = np.mean(feature_matrix,axis=1)
                print(mean_feature.shape)
                std_feature = np.std(feature_matrix, axis=1)
                audio_fvec = np.hstack([mean_feature, std_feature])
                print(audio_fvec.shape)
                audio_features[i] = audio_fvec
                target[i] = label
    return (audio_features, target)

audio_features, target = mfcc(fnames)
audio_features_cqt, target_cqt = mfcc(fnames)

# Linear Support Vector Machine
clf = svm.SVC(gamma='scale', kernel='linear')
clf.fit(audio_features, target)
predicted = cross_val_predict(clf, audio_features, target, cv=10)
conf_mat = metrics.confusion_matrix(target, predicted)
acc_svm = metrics.accuracy_score(target, predicted)
acc_mfcc = acc_svm

print("Confusion matrix:\n%s" % conf_mat)
print("Classification report for classifier %s:\n%s\n"
      % (clf, acc_svm))

clf_cqt = svm.SVC(gamma='scale', kernel='linear')
clf_cqt.fit(audio_features_cqt, target_cqt)
predicted_cqt = cross_val_predict(clf, audio_features_cqt, target_cqt, cv=10)
conf_mat_cqt = metrics.confusion_matrix(target_cqt, predicted_cqt)
acc_cqt = metrics.accuracy_score(target_cqt, predicted_cqt)


acc_vec = conf_mat.diagonal()
map_vec = acc_vec.argsort()
for i in reversed(range(len(map_vec))):
    print("\n", genres[map_vec[i]])

# Naive Bayes Classifier
clf_nb = gnb()
clf_nb.fit(audio_features, target)
predicted_nb = cross_val_predict(clf_nb, audio_features, target, cv=10)
conf_mat_nb = metrics.confusion_matrix(target, predicted_nb)
acc_nb = metrics.accuracy_score(target, predicted_nb)

print("Confusion matrix:\n%s" % conf_mat_nb)
print("Classification report for classifier %s:\n%s\n"
      % (clf_nb, acc_nb))

# Logistic Regression
clf_lr = lr()
clf_lr.fit(audio_features, target)
predicted_lr = cross_val_predict(clf_lr, audio_features, target, cv=10)
conf_mat_lr = metrics.confusion_matrix(target, predicted_lr)
acc_lr = metrics.accuracy_score(target, predicted_lr)

print("Confusion matrix:\n%s" % conf_mat_lr)
print("Classification report for classifier %s:\n%s\n"
      % (clf_lr, acc_lr))

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
train_method = ['SVM', 'Naive bayes', 'Logistic Regression']
accuracy_score = [acc_svm, acc_nb, acc_lr]
ax.bar(train_method, accuracy_score)
plt.show()

"""Based on the classification results, the linear Support Vector Machine performs
   better than the Naive Bayes Classifier. The accuracy of SVM is around 61 percent 
   whereas the accuracy of the Naive Bayes is around 49 percent.
   
   Both the classifiers performed reasonably good for classical, metal and pop. 
   The genres with more rhythm elements like disco, hip-hop and rock had a relatively 
   poor accuracy for Naive bayes"""


# Half Octave Energy

def half_octave_energy(fnames, win_size = 512, hop_length = 512):
    audio_features_2 = np.zeros(len(fnames), 10)
    for (i, fname) in enumerate (fnames):
        half_octave = np.zeros(10)
        for (label,genre) in enumerate(genres):
            if genre in fname:
                sig, s_rate = librosa.load(fname, sr = 22050)
                num_seg = int(len(sig)/hop_length) - 1
                for s in range(num_seg):
                    stft = np.fft.fft(sig[s*hop_length : (s*hop_length)+win_size])

                    #sc = librosa.feature.spectral_centroid(sig[s*hop_length : (s*hop_length)+win_size],
                                                          # s_rate=s_rate, n_fft=win_size, hop_length=hop_length)
                    mag = np.abs(stft)
                    half_octave[0] += mag[0]
                    half_octave[1] += mag[1]
                    for k in range(2,10):
                        half_octave[k] = np.sum(mag[math.pow(2,k-1):math.pow(2,k)])

                # Normalizing the vector
                half_octave = (1 / np.sum(half_octave)) * half_octave

                audio_features_2[i] = half_octave

    return audio_features_2


audio_features_2 = half_octave_energy(fnames, 512, 512)
clf_hoe = svm.SVC(gamma='scale', kernel='linear')
clf_hoe.fit(audio_features_2, target)
predicted_hoe = cross_val_predict(clf_hoe, audio_features_2, target, cv=10)
conf_mat_hoe = metrics.confusion_matrix(target, predicted_hoe)
acc_hoe = metrics.accuracy_score(target, predicted_hoe)

print("Confusion matrix:\n%s" % conf_mat_hoe)
print("Classification report for classifier %s:\n%s\n"
      % (clf_hoe, acc_hoe))


# Bar plot for Features. For CQT
fig2 = plt.figure()
ax2 = fig2.add_axes([0,0,1,1])
feature = ['MFCC', 'CQT', 'Half_Octave']
accuracy_score_2 = [acc_mfcc, acc_cqt, acc_hoe]
ax2.bar(feature, accuracy_score_2)
plt.show()




