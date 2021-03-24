import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob
import librosa
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_predict

fnames = glob.glob("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_visualize/*/*.wav")
genres = ["classical", "hiphop", "country", "rock", "disco", "jazz"]
colours = ['b', 'm', 'y', 'r', 'g', 'c']
audio_features = np.zeros((600, 40))
target = np.zeros(600)


for (i, fname) in enumerate(fnames):
    print("Processing %d %s" % (i, fname))
    for (label, genre) in enumerate(genres):
        if genre in fname:
            audio, srate = librosa.load(fname)
            feature_matrix = librosa.feature.mfcc(y=audio, sr=srate)
            print(feature_matrix.shape)

            mean_feature = np.mean(feature_matrix, axis=1)
            std_feature = np.std(feature_matrix, axis=1)
            audio_fvec = np.hstack([mean_feature, std_feature])
            audio_features[i] = audio_fvec
            target[i] = label

audio_features = MinMaxScaler().fit_transform(audio_features)

def plot_scatter(X, title):

    lw = 2
    for color, i, target_name in zip(colours, [0, 1, 2, 3, 4, 5], genres):
        plt.scatter(X[target == i, 0], X[target == i, 1], color=color, alpha=.8,
                     lw=lw, label=target_name)

    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()

def plot_centroid(X, title):
    lw = 2
    for color, i, target_name in zip(colours, [0, 1, 2, 3, 4, 5], genres):
        plt.scatter(np.average(X[target == i, 0]), np.average(X[target == i, 1]), color=color, alpha=.8,
                    lw=lw, label=target_name)

    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()


X_pca = PCA(n_components=2).fit_transform(audio_features)
plot_scatter(X_pca, '\nPCA')
plot_centroid(X_pca, '\nPCA_Centroid')

tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(audio_features)
plot_scatter(X_tsne, '\nTSNE')
plot_centroid(X_tsne, '\nTSNE_Centroid')

def purity(X):

    first_num = np.zeros(6)
    second_num = np.zeros(6)
    third_num = np.zeros(6)
    fourth_num = np.zeros(6)

    x_var = X[:, 0]
    y_var = X[:, 1]

    for i in [0, 1, 2, 3, 4, 5]:
        x_lab = x_var[target == i]
        y_lab = y_var[target == i]

        for (x,y) in zip(x_lab, y_lab):

            if((x > 0) and (y > 0)):
                first_num[i] += 1

            if((x < 0) and (y > 0)):
                second_num[i] += 1

            if((x < 0) and (y < 0)):
                third_num[i] += 1

            if((x > 0) and (y < 0)):
                fourth_num[i] += 1

    first_purity = np.max(first_num)/np.sum(first_num)
    second_purity = np.max(second_num)/np.sum(second_num)
    third_purity = np.max(third_num)/np.sum(third_num)
    fourth_purity = np.max(fourth_num)/np.sum(fourth_num)

    # Printing the genre labels corresponding to 1st, 2nd, 3rd, 4th quadrants respectively
    print(np.argmax(first_num), np.argmax(second_num), np.argmax(third_num), np.argmax(fourth_num))

    # Printing Purity corresponding to 1st, 2nd, 3rd, 4th quadrants respectively
    print(first_purity, second_purity, third_purity, fourth_purity)

purity(X_pca)
purity(X_tsne)

# Yes the result corresponds to what I perceive visually!!

# Accuracy measure

clf_org = svm.SVC(gamma='scale',kernel='linear')
clf_org.fit(audio_features, target)
predicted_org = cross_val_predict(clf_org, audio_features, target, cv=10)
print("Classification report for classifier using audio features %s:\n%s\n"
      % (clf_org, metrics.accuracy_score(target, predicted_org)))

clf = svm.SVC(gamma='scale',kernel='linear')
clf.fit(X_pca, target)
predicted_pca = cross_val_predict(clf, X_pca, target, cv=10)
print("Classification report for classifier using PCA %s:\n%s\n"
      % (clf, metrics.accuracy_score(target, predicted_pca)))

clf2 = svm.SVC(gamma='scale',kernel='linear')
clf2.fit(X_tsne, target)
predicted_tsne = cross_val_predict(clf2, X_tsne, target, cv=10)
print("Classification report for classifier using tSNE %s:\n%s\n"
      % (clf2, metrics.accuracy_score(target, predicted_tsne)))


