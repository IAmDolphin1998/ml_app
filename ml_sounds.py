# Import packages
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from librosa import stft, amplitude_to_db, load, frames_to_time
from librosa.feature import rms, chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, mfcc
from librosa.display import specshow, waveshow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from ml_utils import fit_model, plot_performances

# Global variables
CATEGORIES = ['happy', 'sad']

########################
##  DATA EXPLORATION  ##
########################
def preprocess_sound(wav_array, wav_sr):
    '''
    Preprocess a sounds for feature exstration.
    '''
    feature_array = []

    # Spectral analysis
    wav_spectral_centroid = spectral_centroid(y = wav_array, sr = wav_sr).mean()
    wav_spectral_bandwidth = spectral_bandwidth(y = wav_array, sr = wav_sr).mean()
    wav_spectral_rolloff = spectral_rolloff(y = wav_array, sr = wav_sr).mean()

    feature_array.append(wav_spectral_centroid)
    feature_array.append(wav_spectral_bandwidth)
    feature_array.append(wav_spectral_rolloff)

    # Wave analysis
    wav_rms = rms(y = wav_array).mean()
    wav_chroma_stft = chroma_stft(y = wav_array, sr = wav_sr).mean()
    wav_zero_crossing_rate = zero_crossing_rate(wav_array).mean()

    feature_array.append(wav_rms)
    feature_array.append(wav_chroma_stft)    
    feature_array.append(wav_zero_crossing_rate)

    # MFCC analysis
    wav_mfccs = mfcc(y = wav_array, sr = wav_sr)
    for wav_mfcc in wav_mfccs:
        feature_array.append(wav_mfcc.mean())

    return feature_array

def make_dataframe(directory, filename):
    '''
    Make DataFrame with sound's features, if necessary.
    '''
    if not os.path.exists(filename):
        # Iterate paths
        sound_paths = []
        for path, _, files in os.walk(directory):
            for name in files:
                sound_paths.append(os.path.join(path, name))
        
        # Iterate sounds
        wav_arrays, wav_srs, wav_labels_id, wav_labels = [], [], [], []
        for sound_path in sound_paths:
            # Get sound's label
            wav_label = [x for x in CATEGORIES if x in sound_path][0]
            wav_label_id = CATEGORIES.index(wav_label)
            
            # Get sound array
            wav_array, wav_sr = load(sound_path, sr = None)
            
            # Append values
            wav_srs.append(wav_sr)
            wav_labels_id.append(wav_label_id)
            wav_labels.append(wav_label)
            
            # Preprocess sound and append features
            wav_array = preprocess_sound(wav_array, wav_sr)        
            wav_arrays.append(wav_array)
        
        df_metadata = pd.DataFrame(list(zip(sound_paths, wav_srs, wav_labels_id, wav_labels)), columns = ['Sound_Path', 'Sample_Rate', 'Class_ID', 'Class'])

        # Build final DataFrame
        df = pd.concat([pd.DataFrame(wav_arrays), df_metadata], axis = 1)

        # Save final DataFrame into XLSX
        df.to_excel(filename)
    else:
        # Load final DataFrame from XLSX
        df = pd.read_excel(filename, index_col = 0)

    return df

def plot_features(ax, wav_feature, name_feature, wav_array, wav_sr):
    '''
    Plot waveform and features of a sound.
    '''
    # Computing the time variable for visualization
    frames = range(len(wav_feature))
    t = frames_to_time(frames, sr = wav_sr)

    # Normalising for visualisation
    wav_feature = minmax_scale(wav_feature, axis = 0)

    #Plotting waveform
    waveshow(wav_array, sr = wav_sr, ax = ax, alpha = 0.4)
    ax.plot(t, wav_feature, color = 'r')

    ax.set_title(name_feature)

df = make_dataframe('.data/sounds', 'my_datasets/sounds.xlsx')

# Class distributions
sns.displot(df.Class, shrink = .8)

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()

# Sample rate distributions
sns.displot(df, x = 'Sample_Rate', hue = 'Class', kind = 'kde')

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()

# Visualize feature extraction
ax_dict = plt.figure(layout = 'constrained').subplot_mosaic(
    [
        ['A0', 'A1'],
        ['B0', 'B1'],
        ['C0', 'C1'],
        ['D0', 'D1']
    ],
)
for i, category in zip(range(2), CATEGORIES):
    audio_path = df[df['Class'] == category].iloc[0].Sound_Path

    # Load audio
    wav_array, wav_sr = load(audio_path, sr = None)

    # Display Spectrogram
    wav_stft = stft(wav_array)
    wav_decibels = amplitude_to_db(abs(wav_stft))
    wav_spectrogram = specshow(wav_decibels, sr = wav_sr, x_axis = 'time', y_axis = 'hz', ax = ax_dict[f'A{i}'])
    ax_dict[f'A{i}'].set_title(f'{category}\nSpectrogram')

    # Displaying MFCCs
    wav_mfccs = mfcc(y = wav_array, sr = wav_sr)
    specshow(wav_mfccs, sr = wav_sr, x_axis = 'time', ax = ax_dict[f'B{i}'])
    ax_dict[f'B{i}'].set_title('MFCC')

    # Display Spectral Centroid
    wav_spectral_centroid = spectral_centroid(y = wav_array, sr = wav_sr)[0]
    plot_features(ax_dict[f'C{i}'], wav_spectral_centroid, 'Spectral Centroid', wav_array, wav_sr)

    # Display Spectral Rolloff
    wav_spectral_rolloff = spectral_rolloff(y = wav_array, sr = wav_sr)[0]
    plot_features(ax_dict[f'D{i}'], wav_spectral_rolloff, 'Spectral Rolloff', wav_array, wav_sr)

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()

##########################
##  DATA PREPROCESSING  ##
##########################

# Drop unnecessary columns and split dataset into train and test
df.drop(['Sound_Path', 'Sample_Rate', 'Class'], axis = 1, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Class_ID'], df.Class_ID, test_size = .25, random_state = 42)

# Scale DataFrames
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)

###########################
##  CLASSIFICATION TASKS ##
###########################

## AdaBoost
adaboost_param_grid = {
    'n_estimators' : list(range(10, 600, 25)),
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1.0],
    'algorithm' : ['SAMME', 'SAMME.R']
}

adaboost_clf = fit_model(GridSearchCV(AdaBoostClassifier(), param_grid = adaboost_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train, y_train, 'my_models/sounds/adaboost.pickle')
plot_performances(adaboost_clf, X_test, y_test, CATEGORIES)

## SVM
svm_param_grid = {
    'C' : [0.1, 1, 10, 100], 
    'gamma' : [1, 0.1, 0.01, 0.001],
    'kernel' : ['rbf', 'sigmoid', 'poly']
}

svm_clf = fit_model(GridSearchCV(SVC(), param_grid = svm_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train_scaled, y_train, 'my_models/sounds/svc.pickle')
plot_performances(svm_clf, X_test_scaled, y_test, CATEGORIES)

## MLP
mlp_param_grid = {
    'hidden_layer_sizes' : [(100), (96, 96, 64, 32), (150, 96, 96, 64, 32, 16)],
    'learning_rate_init' : [0.1, 0.01, 0.001, 0.0001],
    'max_iter' : list(range(200, 1000, 100))
}

mlp_clf = fit_model(GridSearchCV(MLPClassifier(tol = 1e-6), param_grid = mlp_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train, y_train, 'my_models/sounds/cnn.pickle')
plot_performances(mlp_clf, X_test, y_test, CATEGORIES)

## KNN
knn_param_grid = {
    'n_neighbors' : list(range(5, 20, 3)),
    'weights' : ['uniform', 'distance'],
    'metric' : ['euclidean', 'manhattan']
}

knn_clf = fit_model(GridSearchCV(KNeighborsClassifier(), param_grid = knn_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train, y_train, 'my_models/sounds/knn.pickle')
plot_performances(knn_clf, X_test, y_test, CATEGORIES)

##############################
##  ANOMALY DETECTION TASK  ##
##############################

## Isolation Forest
isf_param_grid = {
    'n_estimators': list(range(100, 500, 25)), 
    'max_features': list(range(5, 20, 5))
}

isf_clf = fit_model(GridSearchCV(IsolationForest(), param_grid = isf_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), df.loc[:, df.columns != 'Class_ID'], df.Class_ID, 'my_models/sounds/isf.pickle')

# Calculate anomaly score for each sample
y_score = isf_clf.decision_function(df.loc[:, df.columns != 'Class_ID'])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(df.Class_ID, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(loc = 'lower right')
plt.show()