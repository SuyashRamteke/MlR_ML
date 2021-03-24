import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, metrics
from sklearn.naive_bayes import GaussianNB as gnb
from musicnn.tagger import top_tags



audio_features_df = pd.read_csv('gtzan_audio_features.csv', index_col=0)
filenames_df = pd.read_csv('gtzan_filenames.csv', index_col=0)
tags_df = pd.read_csv('gtzan_MTT_musicnn_tags.csv', index_col=0)
genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'rock', 'blues', 'reggae', 'pop', 'metal']

# Filtering the matrices
audio_features = (audio_features_df.iloc[:, :-1]).to_numpy()
tag_matrix = (tags_df.iloc[:, 0:40]).to_numpy()
print(audio_features[:, :])
print (tags_df.shape)
print(tag_matrix.shape)

# Function for most probable tag
def most_probable_tag():

    for genre in genres:
        genre_df = (tags_df[tags_df['filename'].str.contains(genre)])
        genre_df_sum = (genre_df.iloc[:, 1:-1]).sum()
        top_tag_for_genre = genre_df_sum.idxmax()
        print("Most probable tag for", genre, ":", top_tag_for_genre)


# Function for query by text
tags = (tags_df.columns).tolist()

def query_by_text():
    song = input("Enter the type of song :")
    words = song.split(" ")
    keys = []
    for word in words:
        if word in tags:
            keys.append(word)
    # Creating a copy of tags
    playlist = tags_df
    for key in keys:
        # With every iteration, the key will get modified
        playlist = playlist[playlist[key]==1]

    print(playlist)

# Calling the function three times
query_by_text()
query_by_text()
query_by_text()

# OneVsRest Classifier
clf_SVM = OneVsRestClassifier(svm.SVC(kernel='linear'))
clf_SVM.fit(audio_features, tag_matrix)
predicted_1 = clf_SVM.predict(audio_features)
print("Classification report for classifier %s:\n%s\n"
    %(clf_SVM, metrics.classification_report(tag_matrix, predicted_1)))

clf_Naive_Bayes = OneVsRestClassifier(gnb())
clf_Naive_Bayes.fit(audio_features, tag_matrix)
predicted = clf_Naive_Bayes.predict(audio_features)
print (predicted)
print("Classification report for classifier %s:\n%s\n"
      % (clf_Naive_Bayes, metrics.classification_report(tag_matrix, predicted)))



# Musicnn MTT
tag_1 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/blues/blues.00000.wav", model='MTT_musicnn', topN=5)
tag_2 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/classical/classical.00000.wav", model='MTT_musicnn', topN=5)
tag_3 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/country/country.00000.wav", model='MTT_musicnn', topN=5)
tag_4 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/disco/disco.00000.wav", model='MTT_musicnn', topN=5)
tag_5 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/hiphop/hiphop.00000.wav", model='MTT_musicnn', topN=5)
tag_6 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/jazz/jazz.00000.wav", model='MTT_musicnn', topN=5)
tag_7 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/metal/metal.00000.wav", model='MTT_musicnn', topN=5)
tag_8 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/pop/pop.00000.wav", model='MTT_musicnn', topN=5)
tag_9 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/reggae/reggae.00000.wav", model='MTT_musicnn', topN=5)
tag_10 = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/rock/rock.00000.wav", model='MTT_musicnn', topN=5)

listening_accuracy_mtt = 0.92

# Musicn MSD
tag_1_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/blues/blues.00000.wav", model='MSD_musicnn', topN=5)
tag_2_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/classical/classical.00000.wav", model='MSD_musicnn', topN=5)
tag_3_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/country/country.00000.wav", model='MSD_musicnn', topN=5)
tag_4_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/disco/disco.00000.wav", model='MSD_musicnn', topN=5)
tag_5_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/hiphop/hiphop.00000.wav", model='MSD_musicnn', topN=5)
tag_6_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/jazz/jazz.00000.wav", model='MSD_musicnn', topN=5)
tag_7_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/metal/metal.00000.wav", model='MSD_musicnn', topN=5)
tag_8_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/pop/pop.00000.wav", model='MSD_musicnn', topN=5)
tag_9_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/reggae/reggae.00000.wav", model='MSD_musicnn', topN=5)
tag_10_ = top_tags("/Users/suyashramteke/PycharmProjects/ML_MIR/genres_original/rock/rock.00000.wav", model='MSD_musicnn', topN=5)

listening_accuracy_msd = 0.96