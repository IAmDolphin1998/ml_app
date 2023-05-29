# Import packages
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.ensemble import AdaBoostClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from skimage.io import imread
#from skimage.util import invert
#from skimage.transform import rotate, rescale
from skimage.color import rgb2hsv, rgb2gray, rgb2lab
from skimage.feature import graycomatrix, graycoprops, canny, local_binary_pattern
from ml_utils import fit_model, plot_performances

# Global variables
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

########################
##  DATA EXPLORATION  ##
########################
def preprocess_image(img_array):
    '''
    Preprocess a sounds for feature exstration.
    '''
    feature_array = []

    # Color analysis
    img_array_hsv = rgb2hsv(img_array)
    img_array_lab = rgb2lab(img_array)
    for i in range(3):
        feature_array.append(img_array[:, :, i].mean())
        feature_array.append(img_array[:, :, i].std())

        feature_array.append(img_array_hsv[:, :, i].mean())
        feature_array.append(img_array_hsv[:, :, i].std())
        
        feature_array.append(img_array_lab[:, :, i].mean())
        feature_array.append(img_array_lab[:, :, i].std())

    # Texture analysis
    ccm = graycomatrix((img_array_hsv[:, :, 0] * 360).astype(np.uint16), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels = 360, symmetric = True, normed = True) 
    for f in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        for i in range(4):
            feature_array.append(graycoprops(ccm, f)[0, i])

    img_array_gray = rgb2gray(img_array) 
    glcm = graycomatrix((img_array_gray * 255).astype(np.uint8), [1], [0], levels = 256, symmetric = True, normed = True)
    for f in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        feature_array.append(graycoprops(glcm, f)[0, 0])

    lbp = local_binary_pattern(img_array_gray, 3, 24, 'uniform')
    feature_array.append(lbp.mean())
    feature_array.append(lbp.std())
    
    # Shape analysis
    contours = canny(img_array_gray, sigma = 2)
    y, x = np.nonzero(contours)
    feature_array.append(y.mean())
    feature_array.append(x.mean())

    return feature_array

def make_dataframe(directory, filename):
    '''
    Make DataFrame with image's features, if necessary.
    '''
    if not os.path.exists(filename):
        # Iterate paths
        img_paths = []
        for path, _, files in os.walk(directory):
            for name in files:
                img_paths.append(os.path.join(path, name))
        
        # Iterate images
        img_arrays, img_heights, img_widths, img_labels_id, img_labels = [], [], [], [], []
        for img_path in img_paths:
            # Get image's label
            img_label = [x for x in CATEGORIES if x in img_path][0]
            img_label_id = CATEGORIES.index(img_label)

            # Get image array
            img_array = imread(img_path)
            
            # Append values
            img_heights.append(img_array.shape[0])
            img_widths.append(img_array.shape[1])
            img_labels_id.append(img_label_id)
            img_labels.append(img_label)
            
            # Preprocess image and append features
            img_array = preprocess_image(img_array)           
            img_arrays.append(img_array)
        
        df_metadata = pd.DataFrame(list(zip(img_paths, img_heights, img_widths, img_labels_id, img_labels)), columns = ['Image_Path', 'Height', 'Width', 'Class_ID', 'Class'])

        # Build final DataFrame
        df = pd.concat([pd.DataFrame(img_arrays), df_metadata], axis = 1)

        # Save final DataFrame into XLSX
        df.to_excel(filename)
    else:
        # Load final DataFrame from XLSX
        df = pd.read_excel(filename, index_col = 0)

    return df


df = make_dataframe('.data/images', 'my_datasets/images.xlsx')

# Class distributions
sns.displot(df.Class, shrink = .8)

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()

# Shape distributions
sns.jointplot(data = df, x = 'Height', y = 'Width', hue = 'Class', kind = 'kde')

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()

# Visualize feature extraction
ax_dict = plt.figure(layout = 'constrained').subplot_mosaic(
    [
        ['A0', 'B0'],
        ['A1', 'B1'],
        ['A2', 'B2'],
        ['A3', 'B3'],
        ['A4', 'B4']
    ]
)
for i, category in zip(range(5), CATEGORIES):
    img_path = df[df['Class'] == category].iloc[0].Image_Path

    # Load image
    img_array = imread(img_path)

    # Plot original image
    ax_dict[f'A{i}'].imshow(img_array)
    ax_dict[f'A{i}'].set_axis_off()
    ax_dict[f'A{i}'].set_title(category)

    # Plot all histograms
    ax_dict[f'B{i}'].hist(img_array.ravel(), bins = 256, color = 'orange')
    ax_dict[f'B{i}'].hist(img_array[:, :, 0].ravel(), bins = 256, color = 'red', alpha = .5)
    ax_dict[f'B{i}'].hist(img_array[:, :, 1].ravel(), bins = 256, color = 'green', alpha = .5)
    ax_dict[f'B{i}'].hist(img_array[:, :, 2].ravel(), bins = 256, color = 'blue', alpha = .5)
    ax_dict[f'B{i}'].set_xlabel('Intensity Value')
    ax_dict[f'B{i}'].set_ylabel('Count')
    ax_dict[f'B{i}'].legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel']) 

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show() 

##########################
##  DATA PREPROCESSING  ##
##########################

# Drop unnecessary columns and split dataset into train and test
df.drop(['Image_Path', 'Height', 'Width', 'Class'], axis = 1, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Class_ID'], df.Class_ID, test_size = .25, random_state = 42)

# Scale DataFrames
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)

# Encode targets
y_bin = LabelBinarizer().fit_transform(df.Class_ID)

###########################
##  CLASSIFICATION TASKS ##
###########################

## AdaBoost
adaboost_param_grid = {
    'n_estimators' : list(range(10, 600, 25)),
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1.0],
    'algorithm' : ['SAMME', 'SAMME.R']
}

adaboost_clf = fit_model(GridSearchCV(AdaBoostClassifier(), param_grid = adaboost_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train, y_train, 'my_models/images/adaboost.pickle')
plot_performances(adaboost_clf, X_test, y_test, CATEGORIES)

## SVM
svm_param_grid = {
    'C' : [0.1, 1, 10, 100], 
    'gamma' : [1, 0.1, 0.01, 0.001],
    'kernel' : ['rbf', 'sigmoid', 'poly']
}

svm_clf = fit_model(GridSearchCV(SVC(), param_grid = svm_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train_scaled, y_train, 'my_models/images/svc.pickle')
plot_performances(svm_clf, X_test_scaled, y_test, CATEGORIES)

## MLP
mlp_param_grid = {
    'hidden_layer_sizes' : [(100), (96, 96, 64, 32), (150, 96, 96, 64, 32, 16)],
    'learning_rate_init' : [0.1, 0.01, 0.001, 0.0001],
    'max_iter' : list(range(200, 1000, 100))
}

mlp_clf = fit_model(GridSearchCV(MLPClassifier(tol = 1e-6), param_grid = mlp_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train, y_train, 'my_models/images/mlp.pickle')
plot_performances(mlp_clf, X_test, y_test, CATEGORIES)

## KNN
knn_param_grid = {
    'n_neighbors' : list(range(5, 20, 2)),
    'weights' : ['uniform', 'distance'],
    'metric' : ['euclidean', 'manhattan']
}

knn_clf = fit_model(GridSearchCV(KNeighborsClassifier(), param_grid = knn_param_grid, scoring = 'accuracy', n_jobs = 6, cv = 10, verbose = 2, return_train_score = True), X_train, y_train, 'my_models/images/knn.pickle')
plot_performances(knn_clf, X_test, y_test, CATEGORIES)

##############################
##  ANOMALY DETECTION TASK  ##
##############################

## Isolation Forest
isf_clf = fit_model(OneVsRestClassifier(IsolationForest()), df.loc[:, df.columns != 'Class_ID'], y_bin, 'my_models/images/isf.pickle')

# Calculate anomaly score for each sample
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(len(CATEGORIES)):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], isf_clf.decision_function(df.loc[:, df.columns != 'Class_ID'])[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
colors = ['red', 'blue', 'green', 'yellow', 'magenta']
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color = color, lw = 2, label = f'ROC Curve for class {CATEGORIES[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw = 2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve per class')
plt.legend(loc = 'lower right')

mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()