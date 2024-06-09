import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import multiprocessing

def check_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The following folder does not exist: {folder_path}")
    return folder_path

def get_num_images(subfolders, main_folder):
    num_images = float('inf')
    for subfolder in subfolders:
        current_folder = os.path.join(main_folder, subfolder)
        image_files = [f for f in os.listdir(current_folder) if f.endswith('.bmp')]
        num_images = min(num_images, len(image_files))
    return num_images

def process_folders(main_folder, image_index, num_gray_levels):
    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    all_features = []

    for subfolder in subfolders:
        current_folder = os.path.join(main_folder, subfolder)
        image_files = [f for f in os.listdir(current_folder) if f.endswith('.bmp')]

        if len(image_files) < image_index:
            print(f"Not enough image files in folder: {current_folder}")
            continue

        base_file_name = image_files[image_index]
        full_file_name = os.path.join(current_folder, base_file_name)
        print(f"Now reading {full_file_name}")
        image_array = io.imread(full_file_name)

        if image_array.ndim == 3:
            gray = color.rgb2gray(image_array)
        else:
            gray = image_array

        # Reduce the number of gray levels
        gray = (gray * (num_gray_levels - 1)).astype('uint8')

        glcm = graycomatrix(gray, [1], [0], num_gray_levels, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        entropy_value = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))

        all_features.append([contrast, correlation, energy, homogeneity, entropy_value])

    return np.array(all_features)

def parameters(G, F, model_type):
    data = np.vstack((G, F))
    labels = np.hstack((np.ones(len(G)), -np.ones(len(F))))

    if model_type == 'SVM':
        model = LinearSVC()
    elif model_type == 'LDA':
        model = LDA()
    else:
        raise ValueError("Unknown model type")

    scores = cross_val_score(model, data, labels, cv=5)
    model.fit(data, labels)
    classifier = np.hstack((model.coef_.flatten(), model.intercept_))

    err = 100 * (1 - np.mean(scores))
    return classifier, err, model_type, model.coef_.flatten()

def normalize_data(G, F, ind):
    norm_mean = np.mean(F[:, ind])
    norm_std = np.std(F[:, ind])

    G_norm = G.copy()
    F_norm = F.copy()

    G_norm[:, ind] = (G[:, ind] - norm_mean) / norm_std
    F_norm[:, ind] = (F[:, ind] - norm_mean) / norm_std

    return G_norm, F_norm

def main():
    main_folder1 = 'C:/Users/aria1/OneDrive/Desktop/Master Thesis/Images/GENUINE'
    main_folder2 = 'C:/Users/aria1/OneDrive/Desktop/Master Thesis/Images/HQF'

    main_folder1 = check_folder(main_folder1)
    main_folder2 = check_folder(main_folder2)

    subfolders1 = [f for f in os.listdir(main_folder1) if os.path.isdir(os.path.join(main_folder1, f))]
    subfolders2 = [f for f in os.listdir(main_folder2) if os.path.isdir(os.path.join(main_folder2, f))]

    num_subfolders = min(len(subfolders1), len(subfolders2))
    num_images = min(get_num_images(subfolders1, main_folder1), get_num_images(subfolders2, main_folder2))

    num_gray_levels_list = [16, 32, 64, 128, 256]
    accuracies_dict = {num_levels: np.zeros((num_images, 2)) for num_levels in num_gray_levels_list}
    all_classifiers = []
    normalized_data = []

    def process_image_index(image_index, num_gray_levels):
        all_features1 = process_folders(main_folder1, image_index, num_gray_levels)
        all_features2 = process_folders(main_folder2, image_index, num_gray_levels)

        G = all_features1
        F = all_features2

        print('************  Haralick Features ***************')
        print('SAMPLE SIZE : ')
        print(f'  # OBJECTS : {len(G) + len(F)}')
        print(f'  # FEATURES: {G.shape[1]}')
        print(f' BALANCE : {len(G) / len(F)}')

        print('\nSVM classification:')
        classifier_svm, err_svm, _, vector_svm = parameters(G, F, 'SVM')
        print(f'Feature weights: {classifier_svm[:-1]}')
        print(f'Bias: {classifier_svm[-1]}')
        svm_accuracy = 100 - err_svm

        print('\nLDA classification:')
        classifier_lda, err_lda, _, vector_lda = parameters(G, F, 'LDA')
        print(f'Feature weights: {classifier_lda[:-1]}')
        print(f'Bias: {classifier_lda[-1]}')
        lda_accuracy = 100 - err_lda

        G_norm, F_norm = normalize_data(G, F, 1)

        return {
            'image_index': image_index,
            'num_gray_levels': num_gray_levels,
            'svm_accuracy': svm_accuracy,
            'lda_accuracy': lda_accuracy,
            'classifiers': {'SVM': classifier_svm, 'LDA': classifier_lda},
            'normalized_data': {
                'G_mean': np.mean(G_norm[:, 0]),
                'G_std': np.std(G_norm[:, 0]),
                'F_mean': np.mean(F_norm[:, 0]),
                'F_std': np.std(F_norm[:, 0])
            }
        }

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(process_image_index)(i, num_levels) 
                                         for num_levels in num_gray_levels_list 
                                         for i in range(num_images))

    for result in results:
        image_index = result['image_index']
        num_gray_levels = result['num_gray_levels']
        accuracies_dict[num_gray_levels][image_index, 0] = result['svm_accuracy']
        accuracies_dict[num_gray_levels][image_index, 1] = result['lda_accuracy']
        all_classifiers.append(result['classifiers'])
        normalized_data.append(result['normalized_data'])
        print(f'Image Index: {result["image_index"] + 1}, Gray Levels: {num_gray_levels}, SVM Accuracy: {result["svm_accuracy"]:.2f}%, LDA Accuracy: {result["lda_accuracy"]:.2f}%')

    for num_gray_levels in num_gray_levels_list:
        avg_accuracy_svm = np.mean(accuracies_dict[num_gray_levels][:, 0])
        avg_accuracy_lda = np.mean(accuracies_dict[num_gray_levels][:, 1])
        print(f'Gray Levels: {num_gray_levels}, Average SVM Accuracy: {avg_accuracy_svm:.2f}%, Average LDA Accuracy: {avg_accuracy_lda:.2f}%')

        plt.figure()
        plt.plot(range(1, num_images + 1), accuracies_dict[num_gray_levels][:, 0], '-o', label=f'SVM Accuracy ({num_gray_levels} levels)')
        plt.plot(range(1, num_images + 1), accuracies_dict[num_gray_levels][:, 1], '-x', label=f'LDA Accuracy ({num_gray_levels} levels)')
        plt.xlabel('Image Index')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy for Each Image Index (Gray Levels: {num_gray_levels})')
        plt.legend()
        plt.savefig(f'AccuracyGraph_{num_gray_levels}.png')
        plt.show()

    np.save('accuracies.npy', accuracies_dict)
    np.save('all_classifiers.npy', all_classifiers)
    np.save('normalized_data.npy', normalized_data)

if __name__ == "__main__":
    main()
