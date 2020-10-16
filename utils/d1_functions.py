from collections import defaultdict
import pandas as pd
import numpy as np
from itertools import product
# from sklearn.metrics import confusion_matrix, accuracy_score, \
# precision_recall_fscore_support
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

# def report_performance(y_true, y_predicted, labels):
#     """Calculates a set of performance measures in a classification 
#     problem. Measure are the following: accuracy, precision, recall, 
#     f1-score. Additionally it reports the support of each class.
    
#     ARGUMENTS:
#     y_true -> np.array, vector of true classes
#     y_predicted -> np.array, vector of predicted classes
#     labels -> list of str, names of each class
#     """
    
#     # initialize data structure 
#     performance_dict =defaultdict(lambda: [])

#     # accuracy
#     performance_dict['accuracy'].extend(
#         [accuracy_score(y_true, y_predicted)] * len(labels))

#     # precision, recall, f1-score and support
#     pres, rec, f1, sup = precision_recall_fscore_support(y_true, y_predicted)

#     ## precision
#     performance_dict['precision'].extend(pres)

#     ## recall
#     performance_dict['recall'].extend(rec)

#     ## f1-score
#     performance_dict['f1-score'].extend(f1)
    
#     ## support
#     performance_dict['support'].extend(sup)
    
#     # build df
#     performance_df = pd.DataFrame(performance_dict, index=labels)
    
#     # calculate average
#     performance_df.loc['AVERAGE'] = performance_df.mean(axis=0)
    
#     # transform support colum in integer
#     performance_df.loc[:, 'support'] = performance_df['support'].round(
#         0).astype(int)

#     return performance_df * 100

def plot_city_prediction_ground_truth(city_name, path_db_lab = "data/cities_arrays/", 
    lab_sx = "_l-01243567.npy", show_bb = False, path_db_img = "data/cities_arrays/", 
    path_db_pdt = "data/cities_arrays/", img_sx = "_Lsat.npy"):
    if not show_bb:
        # color map
        # l_colors = ["#ffffff", '#1986d5','#1dcea3','#2ec421', '#fffa00','#f52806','#ed0b81', '#d90fe7','#7013e1','#161ddc']
        l_colors = ["#2a2a2a","#FFFFFF", "#00cc00", "#F4F60C", "#1e83c3"]
        cmapa1 = colors.ListedColormap(l_colors)
        

        # load array
        pdt_array = np.load(path_db_pdt + city_name + '_predicted.npy') + 1
        lab_array = np.load(path_db_lab + city_name + lab_sx)
        # print(np.unique(pdt_array), np.unique(lab_array))
        color_dict = {}
        for i, j in product(range(1,4), range(1,4)):
            if i == j: 
                color_dict[(i, j)] = i
            else:
                color_dict[(i, j)] = -1
        for j in range(1,4):
            color_dict[(0, j)] = 0
        # print(color_dict)

        # transform labels
        UF_config = dict(zip([4,5,6,1,2,3,7,0], [1,1,1,2,2,2,3,0]))
        transform = lambda x: UF_config[x]
        v_transform = np.vectorize(transform)
        lab_array = v_transform(lab_array)

        # color assignation
        assign = lambda x, y: color_dict[(x,y)]
        v_assign = np.vectorize(assign)
        array = v_assign(lab_array, pdt_array)

        # print(np.unique(array, return_counts = True))

        # plot parameters
        fig = plt.imshow(array, cmap = cmapa1)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        return(fig, l_colors)
    else:
        # color map
        l_colors1 = ["#2a2a2a","#FFFFFF", "#00cc00", "#F4F60C", "#1e83c3"]
        l_colors2 = ["#FFFFFF", "#00cc00", "#F4F60C", "#1e83c3"]

        # load array
        array = np.load(path_db_lab + city_name + lab_sx)
        sat_array = np.load(path_db_img + city_name + img_sx)
        i_x, i_y = np.nonzero(sat_array[:,:,0]==0)
        array[i_x, i_y ] = -1

        if len(i_x) > 0:
            cmapa1 = colors.ListedColormap(l_colors1)
        else:
            cmapa1 = colors.ListedColormap(l_colors2)
        return(plt.imshow(array, cmap = cmapa1), l_colors1)

def plot_city_prediction(city_name, show_bb = False, path_db_img = "data/cities_arrays/", 
    path_db_pdt = "data/cities_arrays/", img_sx = "_Lsat.npy"):
    # if not show_bb:
        # color map

    l_colors = ["#2a2a2a", "#00cc00", "#F4F60C", "#1e83c3"]
    cmapa1 = colors.ListedColormap(l_colors)
    

    # load array
    pdt_array = np.load(path_db_pdt + city_name + '_predicted.npy') + 1

    if show_bb:
        sat_array = np.load(path_db_img + city_name + '_Lsat.npy')
        i_x, i_y = np.nonzero(sat_array[:,:,0]==0)
        pdt_array[i_x, i_y ] = 0
    
    # plot parameters
    fig = plt.imshow(pdt_array, cmap = cmapa1)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    return(fig, l_colors)
    # else:
    #     pass



# def calculate_plot_confusion_matrix(city_name, class_names,
#                           normalize=True,
#                           path_db_lab = "data/cities_arrays/",
#                           path_db_pdt = "data/cities_arrays/",
#                           lab_sx = "_l-01243567.npy",
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues,
#                           text_size=16):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     # calculate CM
#     pdt_array = np.load(path_db_pdt + city_name + '_predicted.npy') + 1
#     lab_array = np.load(path_db_lab + city_name + lab_sx)
#     pdt_array = pdt_array[np.nonzero(lab_array)]
#     lab_array = lab_array[np.nonzero(lab_array)]
#     # transform labels
#     UF_config = dict(zip([4,5,6,1,2,3,7,0], [1,1,1,2,2,2,3,0]))
#     transform = lambda x: UF_config[x]
#     v_transform = np.vectorize(transform)
#     lab_array = v_transform(lab_array)
#     cm = confusion_matrix(lab_array, pdt_array, labels = [1, 2, 3]) 


    # plot CM
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # transform to percentage
    cm = cm * 100

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=text_size)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=text_size)
    #cb.ax.set_aspect(6)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=text_size)
    plt.yticks(tick_marks, class_names, fontsize=text_size)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 1),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=text_size)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=text_size)
    plt.xlabel('Predicted label', fontsize=text_size)
    # print(cm)


def plot_prediction_labels(ax, colores):
    # définir les ticks invisibles
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fntsize = 11

    # légende
    patches = [mpatches.Patch(color=colores[1], label="predicted open land"),
               mpatches.Patch(color=colores[2], label="predicted built-up"),
               mpatches.Patch(color=colores[3], label="predicted water")]
    plt.legend(handles=patches,  loc='lower right',
               borderaxespad=.5, fontsize=fntsize);