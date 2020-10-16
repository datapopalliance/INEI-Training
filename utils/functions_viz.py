# FunctionsViz
#from osgeo import gdal
#import matplotlib as mpl
#mpl.rc('font',family='Times') # does not work on Ubuntu
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import itertools as it
import os
# from sklearn import metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
#import os
#from FunctionsLandsat import *

def saveresults(filename, report, accuracy, parameters,cities):
    with open(filename, 'a') as file:
        file.write("=======================================\n\n\n")
        file.write('cities: ')
        file.write(str(cities)+'\n')
        file.write(parameters)
        file.write(report)
        file.write('\n' + accuracy)
        file.write('\n\n\n')
        file.write("=======================================\n")

filename = 'reports/results.txt'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    print("perro")
    if normalize == "recall":
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],3)
    elif normalize == "precision":
        cm = np.round(cm.astype('float') / cm.sum(axis=0),3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def plot_city_labels(img_name, show_bb = True, 
    path_db_lab = "../db/lab/NYU-UF/", lab_sx = "_l-01243567.npy",  
    path_db_img = "../db/img/UF_unscaled_bbox/",
    img_sx = "_Lsat.npy", map_labels=False):
    '''
    Plots a the labels of a satellite image.

    Parameters
    ----------
    img_name : str,
        name of the image to be plotted. Available choices are: [
        `BeloHorizonte`, `Bogota`, `BuenosAires`, `Cabimas`, `Caracas`, 
        `Cochabamba2`, `Cochabamba`, `Cordoba`, `Culiacan2`, `Culiacan`,
        `Curitiba2`, `Curitiba`, `Florianopolis`, `Guadalajara`, 
        `Guatemala`, `Holguin`, `Ilheus2`, `Ilheus`, `Jequie`, `Leon`, 
        `Mexico2`, `Mexico`, `Palmas`, `Quito`, `Reynosa`, `Ribeirao`, 
        `SanSalvador2`, `SanSalvador`, `Santiago`, `SaoPaulo2`, `SaoPaulo`, 
        `Tijuana2`, `Tijuana`, `Valledupar`].

    show_bb: bool,
        whether the black bands of the satellite images, or segmets
        captured by other images (in case of multi image cities) are 
        plotted.

    path_db_lab : str,
        path to the directory where all label arrays are stored. 

    lab_sx : str,
        sufix used in every filename of the label files. 

    path_db_img : str,
        path to the directory where all images arrays are stored. 
        Ignored if `show_bb` is False.

    img_sx : str,
        suffix used in every filename of the array files. Ignored if 
        `show_bb` is False.

    map_labels : bool,
        whether original labels are maped to 3 classes used in the 
        project: (1) built-up, (2) open space and (3) water. Defaults to 
        False.
    '''
    

     # load array
    array = np.load(path_db_lab + img_name + lab_sx)

    
    # color map
    if map_labels:
        l_colors = ["#FFFFFF", "#F4F60C", "#00cc00", "#1e83c3"]
    else:
        l_colors = ["#FFFFFF", "#F4F60C", "#F4FF9C", "#A4FF4C", "#00FF66",
            "#ccFFcc", "#00cc00", "#1e83c3"]
    cmapa = colors.ListedColormap(l_colors)

    # define legend
    if map_labels:
        patches = [mpatches.Patch(color=l_colors[1], label="built-up"),
                   mpatches.Patch(color=l_colors[2], label="open land"),
                   mpatches.Patch(color=l_colors[3], label="water")]
    else:
        patches = [
            mpatches.Patch(color=l_colors[1], label="urban built-up"),
            mpatches.Patch(color=l_colors[2], label="suburban built-up"),
            mpatches.Patch(color=l_colors[3], label="rural built-up"),
            mpatches.Patch(color=l_colors[4], label="urbanized open land"),
            mpatches.Patch(color=l_colors[5], label="captured open land"),
            mpatches.Patch(color=l_colors[6], label="rural open land"),
            mpatches.Patch(color=l_colors[7], label="water")
        ]


    # transform labels
    if map_labels:
        label_transform = {1:[1,2,3], 2:[4,5,6], 3:[7]}

        for k in label_transform:

            for v in label_transform[k]:

                array[array == v] = k

    else:
        pass # need to be built

    # handle black bands option
    if show_bb:

        # load array
        sat_array = np.load(path_db_img + img_name + img_sx)
        i_x, i_y = np.nonzero(sat_array[:,:,0]==0)
        array[i_x, i_y ] = -1


        if len(i_x) > 0:

            # redefine color map
            l_colors = ["#2a2a2a"] + l_colors
            cmapa = colors.ListedColormap(l_colors)
            patches = patches + [mpatches.Patch(color=l_colors[0], label="black band")]


    fig, ax = plt.subplots(figsize=(16,8))

    # plot image
    plt.imshow(array, cmap=cmapa)
    
    #plt parameters
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # plot legend
    fntsize = 11
    plt.legend(handles=patches,  loc='best',
               borderaxespad=.5, fontsize=fntsize);

    return(fig)
    

    

def plot_city_spectrum(city_name, l_colors, path_db_lab = "db/lab/NYU-UE_FMw/", 
    lab_sx = "_l-0123.npy", show_bb = True, path_db_img = "db/img/unscaled_bbox-UE/",
    img_sx = "_Lsat.npy"):

    #["#00cc00", "#F4F60C", "#1e83c3"]

    # load array
    array = np.load(path_db_lab + city_name + lab_sx)
    n_classes = array.max()
    p(n_classes)
    sat_array = np.load(path_db_img + city_name + img_sx)
    n_bands = sat_array.shape[2]
    print(sat_array.shape)
    i_x, i_y = np.nonzero(sat_array[:,:,0]==0)
    array[i_x, i_y ] = -1

    # plot it
    fig, axarr = plt.subplots(n_classes,1, figsize=(24, 30))
    fig.suptitle(city_name, size=fntsize*2)
    
    # plot spctrum bars

    # size parameters
    fnt_size = 14
    fnt_size_2 = 12
    width = 0.4  # the width of the bars: can also be len(x) sequence

    # plot values
    means = np.array([np.array([np.mean(sat_array[:, :, j][array == i]) for i in range(1, n_classes + 1)]) 
                      for j in range(0, n_bands)])
    stds = np.array([np.array([np.std(sat_array[:, :, j][array == i]) for i in range(1, n_classes + 1)]) 
                     for j in range(0, n_bands)])
    #print(means.shape, stds.shape)
    class_names = ["Non built-up", "Urban built-up", "Water"]

    # plot
    for k in range(n_classes):
        pos_samples = np.nonzero(array == k + 1)
        choice = np.random.choice(len(pos_samples[0]), 25, replace=False)
        pos_samples = [arr[choice] for arr in pos_samples]
        sample = sat_array[pos_samples]
        for l in range(25):
            axarr[k].plot(range(1,11), sample[l, :], l_colors[k])

        axarr[k].plot(range(1,11), means[:, k],'k', linewidth=10)
        axarr[k].plot(range(1,11), means[:, k] + stds[:, k], '--k', linewidth=6)
        axarr[k].plot(range(1,11), means[:, k] - stds[:, k], '--k', linewidth=6)
        axarr[k].set_ylabel('Value', labelpad=0, size=fnt_size_2*2)
        axarr[k].set_xlabel('Band', labelpad=0, size=fnt_size_2*2)
        axarr[k].set_title(class_names[k] + ' - Spectrum', size=fnt_size*2)
        axarr[k].set_xticks(range(1, n_bands + 1))#, size=fnt_size_2)
        axarr[k].tick_params(axis='both', labelsize=fnt_size_2*2)


    return(fig)

def plot_city_prediction(city_name, classif_name, path_db_lab = "db/lab/NYU-UF/", 
    lab_sx = "_l-01243567.npy", show_bb = False, path_db_img = "db/img/UF_unscaled_bbox/",
    img_sx = "_Lsat.npy"):
    if not show_bb:
        # color map
        # l_colors = ["#ffffff", '#1986d5','#1dcea3','#2ec421', '#fffa00','#f52806','#ed0b81', '#d90fe7','#7013e1','#161ddc']
        l_colors = ["#2a2a2a","#FFFFFF", "#00cc00", "#F4F60C", "#1e83c3"]
        cmapa1 = colors.ListedColormap(l_colors)
        path_db_pdt = "db/lab/{}/".format(classif_name)

        # load array
        pdt_array = np.load(path_db_pdt + city_name + '.npy')
        lab_array = np.load(path_db_lab + city_name + lab_sx)
        color_dict = {}
        for i, j in it.product(range(1,4), range(1,4)):
            if i == j: 
                color_dict[(i, j)] = i
            else:
                color_dict[(i, j)] = -1
        for j in range(1,4):
            color_dict[(0, j)] = 0
        print(color_dict)

        # transform labels
        UF_config = dict(zip([4,5,6,1,2,3,7,0], [1,1,1,2,2,2,3,0]))
        transform = lambda x: UF_config[x]
        v_transform = np.vectorize(transform)
        lab_array = v_transform(lab_array)

        # color assignation
        assign = lambda x, y: color_dict[(x,y)]
        v_assign = np.vectorize(assign)
        array = v_assign(lab_array, pdt_array)

        print(np.unique(array, return_counts = True))

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


def calculate_plot_confusion_matrix(city_name, class_names,
                          classif_name,
                          normalize=True,
                          path_db_lab = "db/lab/NYU-UF/",
                          lab_sx = "_l-01243567.npy",
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          text_size=16):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # calculate CM
    path_db_pdt = "db/lab/{}/".format(classif_name)
    pdt_array = np.load(path_db_pdt + city_name + '.npy')
    lab_array = np.load(path_db_lab + city_name + lab_sx)
    pdt_array = pdt_array[np.nonzero(lab_array)]
    lab_array = lab_array[np.nonzero(lab_array)]
    # transform labels
    UF_config = dict(zip([4,5,6,1,2,3,7,0], [1,1,1,2,2,2,3,0]))
    transform = lambda x: UF_config[x]
    v_transform = np.vectorize(transform)
    lab_array = v_transform(lab_array)
    cm = metrics.confusion_matrix(lab_array, pdt_array, labels = [1, 2, 3]) 


    # plot CM
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm = cm * 100

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=text_size)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=text_size)
    #cb.ax.set_aspect(6)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=text_size)
    plt.yticks(tick_marks, class_names, fontsize=text_size)

    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=text_size)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=text_size)
    plt.xlabel('Predicted label', fontsize=text_size)

def plot_single_band(img_name, band,
    path_db_img = "db/img/UF_unscaled_bbox",
    img_sx = "_Lsat.npy", color_scale='gray'):
    '''Plots a single band of the satellite image in a colorscale.

    Parameters
    ----------
    img_name : str,
        name of the image to be plotted. Available choices are: [
        `BeloHorizonte`, `Bogota`, `BuenosAires`, `Cabimas`, `Caracas`, 
        `Cochabamba2`, `Cochabamba`, `Cordoba`, `Culiacan2`, `Culiacan`,
         `Curitiba2`, `Curitiba`, `Florianopolis`, `Guadalajara`, 
         `Guatemala`, `Holguin`, `Ilheus2`, `Ilheus`, `Jequie`, `Leon`, 
         `Mexico2`, `Mexico`, `Palmas`, `Quito`, `Reynosa`, `Ribeirao`, 
         `SanSalvador2`, `SanSalvador`, `Santiago`, `SaoPaulo2`, `SaoPaulo`, 
         `Tijuana2`, `Tijuana`, `Valledupar`].

    band : int,
        number of the band to be plotted. Choices range from 1 to 11.

    path_db_img : str,
        path to the directory where all images arrays are stored. 

    img_sx : str,
        suffix used in every filename of the array files.

    cmap : str,
        used as argument of `matplotlib.pyplot.get_cmap`. Color scale used 
        for plotting the band image.

    '''

    # color map
    color_map = plt.get_cmap(color_scale)

    # load array
    sat_array = np.load(path_db_img + '/' + img_name + img_sx)
    i_x, i_y = np.nonzero(sat_array[:,:,0]==0)

    
    # plot image
    fig, ax = plt.subplots(figsize=(16,8))
    plt.imshow(sat_array[:, :, band - 1], cmap=color_map, vmin=0, 
        vmax=sat_array.max() * 0.5)

    #plt parameters
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    

    return(fig)

def plot_band_combination(
    img_name, bands_list=[4, 3, 2],
    path_db_img = "db/img/UF_unscaled_bbox", img_sx = "_Lsat.npy", 
    saturation_coeff=1):
    '''
    Plots a combinations of 3 bands of a satellite image as it was 
    RGB bands.
    
    Parameters
    ----------
    img_name : str,
        name of the image to be plotted. Available choices are: [
        `BeloHorizonte`, `Bogota`, `BuenosAires`, `Cabimas`, `Caracas`, 
        `Cochabamba2`, `Cochabamba`, `Cordoba`, `Culiacan2`, `Culiacan`,
        `Curitiba2`, `Curitiba`, `Florianopolis`, `Guadalajara`, 
        `Guatemala`, `Holguin`, `Ilheus2`, `Ilheus`, `Jequie`, `Leon`, 
        `Mexico2`, `Mexico`, `Palmas`, `Quito`, `Reynosa`, `Ribeirao`, 
        `SanSalvador2`, `SanSalvador`, `Santiago`, `SaoPaulo2`, `SaoPaulo`, 
        `Tijuana2`, `Tijuana`, `Valledupar`].
        
    bands_list: list of integers,
        indicates which bands will be mapped to RGB (respecting the order).
        
    path_db_img : str,
        path to the directory where all images arrays are stored. 

    img_sx : str,
        suffix used in every filename of the array files.
        
    saturation_coeff : float,
        between 0 and 1. Controls saturation of the plotted image, if 1 
        saturation value will be number considered as the maximum value 
        each band can attain. If 0, it will be the minimum, thus the image 
        will be completely white. Defaults to 1.
        
    '''
    
    # load array
    sat_array = np.load(path_db_img + '/' + img_name + img_sx)
    img_array = sat_array[:, :, np.array(bands_list) - 1]
    img_array = (img_array - 1) / min(65535, saturation_coeff * 65535)

    img_array[img_array > 1] = 1

    # plot image
    fig, ax = plt.subplots(figsize=(16,8))
    plt.imshow(img_array, vmin=0, 
        vmax=sat_array.max() * 0.5)

    #plt parameters
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    return(fig)