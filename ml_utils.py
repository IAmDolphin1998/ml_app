import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

def plot_classification_report(ax, classification_report_dict, CATEGORIES):
    '''
    Plot scikit-learn classification report.
    '''
    bar_width = .25
    xticklabels = []
    bar_colors = ['blue', 'orange', 'green']
    for i, category in enumerate(CATEGORIES, start = 1):
        sup = classification_report_dict[category].pop('support')

        for j, (attribute, measurement) in enumerate(classification_report_dict[category].items()):
            rects = ax.bar(i + j * bar_width, measurement, bar_width, color = bar_colors[j], label = attribute)
            ax.bar_label(rects, labels = [f'{measurement:.2%}'], padding = 3)

        xticklabels.append(f'{category} ({sup})')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Classification report')
    ax.set_xticks(np.arange(1, len(xticklabels) + 1) + bar_width, xticklabels)
    ax.legend(['precision', 'recall', 'f1-score'], ncols = 3)

def plot_performances(clf, X_test, y_test, CATEGORIES):
    '''
    Plot scikit-learn model performances.
    '''
    # Build figure
    ax_dict = plt.figure(layout = 'constrained').subplot_mosaic(
        [
            ['A', 'A', 'D', 'B'],
            ['A', 'A', 'C', 'C'],
        ]
    )

    # Plot CV results
    cv_results_ = pd.DataFrame(clf.cv_results_)
    filtered_cv_results_ = cv_results_.filter(regex = 'split.*_train_.*')
    
    for index, row in filtered_cv_results_.iterrows():
        if index != clf.best_index_:
            ax_dict['A'].plot(row.values, 'o--b')
        else:
            ax_dict['A'].plot(row.values, 'o-r', linewidth = 3.0)
    ax_dict['A'].set_title('Grid search scores')
    ax_dict['A'].set_ylabel('Train score')
    ax_dict['A'].set_xlabel('N° fold')

    # Predict test samples
    y_pred = clf.predict(X_test)

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax = ax_dict['B'])
    ax_dict['B'].set_title('Confusion Matrix')

    # Plot classification report
    classification_report_dict = classification_report(y_test, y_pred, target_names = CATEGORIES, output_dict = True)
    plot_classification_report(ax_dict['C'], classification_report_dict, CATEGORIES)

    # Add text informations
    textstr_tex = '\\textbf{Best params}'
    for key, value in clf.best_params_.items():
        textstr_tex += f'\n\n{key} : {value}'
    textstr_tex += '\n\n\\textbf{Metrics}'
    textstr_tex += f"\n\nAccuracy : {classification_report_dict['accuracy']:.2%}"

    ax_dict['D'].text(0.2, 0.2, textstr_tex, transform = ax_dict['D'].transAxes, fontsize = 14, usetex = True, bbox = {'boxstyle' : 'round', 'facecolor' : 'wheat', 'alpha' : 0.5})
    ax_dict['D'].set_axis_off()

    plt.suptitle(f'Performances report for {clf.best_estimator_.__class__.__name__}', fontsize = 20)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()

def fit_model(clf, X, y, filename):
    '''
    Fit scikit_learn model, if necessary.
    '''
    if not os.path.exists(filename):
        # Fit model
        clf.fit(X, y)

        # Save model
        pickle.dump(clf, open(filename, 'wb'))
    else:
        # Load model
        clf = pickle.load(open(filename, 'rb'))
    
    return clf