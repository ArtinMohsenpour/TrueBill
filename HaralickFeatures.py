import os
import numpy as np
import skimage.io as io
import skimage.color as color
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pickle

def main():
    # Initialize main folder paths
    main_folder1 = r'C:\Users\aria1\OneDrive\Desktop\Master Thesis\Images\GENUINE'
    main_folder2 = r'C:\Users\aria1\OneDrive\Desktop\Master Thesis\Images\HQF'

    # Check if folders exist
    main_folder1 = check_folder(main_folder1)
    main_folder2 = check_folder(main_folder2)

    # Get the number of subfolders to determine the number of images to process
    subfolders1 = [f.path for f in os.scandir(main_folder1) if f.is_dir()]
    subfolders2 = [f.path for f in os.scandir(main_folder2) if f.is_dir()]
    num_subfolders = min(len(subfolders1), len(subfolders2))

    # Assume all subfolders have the same number of images
    num_images = min(get_num_images(subfolders1), get_num_images(subfolders2))

    accuracies = np.zeros((num_images, 2))  # Store accuracies for SVM and LDA
    all_classifiers = []
    normalized_data = []

    for image_index in range(num_images):
        # Process folders and calculate features for the current image index
        all_features1 = process_folders(main_folder1, 1, image_index)
        all_features2 = process_folders(main_folder2, 1, image_index)

        G = all_features1
        F = all_features2

        print('************  Haralick Features ***************')
        print(f'SAMPLE SIZE : ')
        print(f'  # OBJECTS : {len(np.concatenate([G[:, 0], F[:, 0]]))}')
        print(f'  # FEATURES: {len(G[0, :])}')
        print(f' BALANCE : {len(G[:, 0]) / len(F[:, 0])}')

        # SVM classification
        print('\nSVM classification:')
        classifier_svm, err_svm, vector_svm = parameters(G, F, 'SVM')
        print(f'Feature weights: {classifier_svm[:-1]}')
        print(f'Bias: {classifier_svm[-1]}')
        D_SVM = discriminate_dim(G, F, classifier_svm[:-1])
        print(f'Accuracy of SVM: {100 - err_svm}%')
        print(f'Discriminant of feature combination within SVM weights: {D_SVM}')

        accuracies[image_index, 0] = 100 - err_svm
        all_classifiers.append({'SVM': classifier_svm})

        # LDA classification
        print('\nLDA classification:')
        classifier_lda, err_lda, vector_lda = parameters(G, F, 'LDA')
        print(f'Feature weights: {classifier_lda[:-1]}')
        print(f'Bias: {classifier_lda[-1]}')
        D_LDA = discriminate_dim(G, F, classifier_lda[:-1])
        print(f'Accuracy of LDA: {100 - err_lda}%')
        print(f'Discriminant of feature combination within LDA weights: {D_LDA}')

        accuracies[image_index, 1] = 100 - err_lda
        all_classifiers[image_index]['LDA'] = classifier_lda

        # Normalize data and save statistics
        G_norm, F_norm = normalize_data(G, F, 0)
        normalized_data.append({
            'G_mean': np.mean(G_norm[:, 0]),
            'G_std': np.std(G_norm[:, 0]),
            'F_mean': np.mean(F_norm[:, 0]),
            'F_std': np.std(F_norm[:, 0]),
        })

        print(f'Image Index: {image_index}, SVM Accuracy: {100 - err_svm:.2f}%, LDA Accuracy: {100 - err_lda:.2f}%')

    # Calculate and display average accuracies
    avg_accuracy_svm = np.mean(accuracies[:, 0])
    avg_accuracy_lda = np.mean(accuracies[:, 1])
    print(f'Average SVM Accuracy: {avg_accuracy_svm:.2f}%')
    print(f'Average LDA Accuracy: {avg_accuracy_lda:.2f}%')

    # Save useful information to files
    with open('accuracies.pkl', 'wb') as f:
        pickle.dump(accuracies, f)
    with open('all_classifiers.pkl', 'wb') as f:
        pickle.dump(all_classifiers, f)
    with open('normalized_data.pkl', 'wb') as f:
        pickle.dump(normalized_data, f)

    # Generate and save graphs
    plt.figure()
    plt.plot(range(num_images), accuracies[:, 0], '-o', label='SVM Accuracy')
    plt.plot(range(num_images), accuracies[:, 1], '-x', label='LDA Accuracy')
    plt.xlabel('Image Index')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy for Each Image Index')
    plt.legend()
    plt.savefig('AccuracyGraph.png')

def check_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f'Error: The following folder does not exist:\n{folder_path}')
    return folder_path

def get_num_images(subfolders):
    num_images = float('inf')
    for folder in subfolders:
        image_files = [f for f in os.listdir(folder) if f.endswith('.bmp')]
        num_images = min(num_images, len(image_files))
    return num_images

def process_folders(main_folder, num_images_per_folder, image_index):
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    num_subfolders = len(subfolders)
    all_features = np.zeros((num_subfolders * num_images_per_folder, 5))

    for k, folder in enumerate(subfolders):
        image_files = [f for f in os.listdir(folder) if f.endswith('.bmp')]
        if len(image_files) < image_index + 1:
            print(f'Not enough image files in folder: {folder}')
            continue

        full_file_name = os.path.join(folder, image_files[image_index])
        print(f'Now reading {full_file_name}')
        image_array = io.imread(full_file_name)

        gray = color.rgb2gray(image_array) if image_array.ndim == 3 else image_array
        glcm = greycomatrix(gray.astype(np.uint8), [1], [0], symmetric=True, normed=True)
        stats = {prop: greycoprops(glcm, prop)[0, 0] for prop in ('contrast', 'correlation', 'energy', 'homogeneity')}
        entropy_value = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))

        row_index = k * num_images_per_folder
        all_features[row_index, :] = [stats['contrast'], stats['correlation'], stats['energy'], stats['homogeneity'], entropy_value]

    return all_features

def parameters(G, F, classifier_type):
    data = np.vstack((G, F))
    labels = np.hstack((np.ones(G.shape[0]), -1 * np.ones(F.shape[0])))

    if classifier_type == 'SVM':
        model = svm.SVC(kernel='linear')
        scores = cross_val_score(model, data, labels, cv=5)
        model.fit(data, labels)
        classifier = np.append(model.coef_, model.intercept_)
        err = 100 * (1 - np.mean(scores))
        vector = model.coef_.flatten()
    elif classifier_type == 'LDA':
        model = LinearDiscriminantAnalysis()
        scores = cross_val_score(model, data, labels, cv=5)
        model.fit(data, labels)
        coef = model.coef_.flatten()
        intercept = model.intercept_.flatten()
        classifier = np.append(coef, intercept)
        err = 100 * (1 - np.mean(scores))
        vector = coef
    else:
        raise ValueError('Unknown classifier type')

    return classifier, err, vector

def normalize_data(G, F, ind):
    norm_mean = np.mean(F[:, ind])
    norm_std = np.std(F[:, ind])

    G_norm = np.copy(G)
    F_norm = np.copy(F)

    G_norm[:, ind] = (G[:, ind] - norm_mean) / norm_std
    F_norm[:, ind] = (F[:, ind] - norm_mean) / norm_std

    return G_norm, F_norm

def discriminate_dim(G, F, weights):
    return abs(np.mean(G @ weights) - np.mean(F @ weights)) / np.sqrt(np.var(G @ weights) + np.var(F @ weights))

if __name__ == '__main__':
    main()
