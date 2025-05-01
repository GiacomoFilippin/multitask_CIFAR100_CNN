import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_data_dir(subfolder_name=""):
    try:
        # Assumes the script is either in the 'main_code' directory or in the "scripts" one
        script_dir = os.path.dirname(__file__)
    except NameError:
        # Fallback if __file__ is not defined (common in notebooks/interactive)
        # Assumes the current working directory is 'main_code' or 'OPT_scriptshw'
        script_dir = os.getcwd()
        # If your notebook/script is actually in OPT_hw, adjust the path construction:
        # data_dir = os.path.abspath(os.path.join(script_dir, 'data', 'raw', 'archive'))
    # Construct the path relative to the parent directory of script_dir
    # Goes up one level from script_dir (main_code) to OPT_hw, then into data/raw/archive
    data_dir_relative = os.path.join(script_dir, '..', 'data', 'raw', subfolder_name)

    # Get the absolute, normalized path (resolves '..')
    data_dir = os.path.abspath(data_dir_relative)

    print(f"Data directory path: {data_dir}")
    return data_dir

def load_cifar_datafile(folder_path, filename):
    """
    Carica un singolo batch del dataset CIFAR-100 da un file specificato.

    Args:
        folder_path (str): Il percorso della cartella contenente il file.
        filename (str): Il nome del file del batch da caricare (es. 'train', 'test', 'meta').

    Returns:
        dict: Un dizionario contenente i dati del batch, oppure None se il file non viene trovato o si verifica un errore.
    """
    file_path = os.path.join(folder_path, filename)
    print(f"Tentativo di caricamento da: {file_path}")
    try:
        with open(file_path, 'rb') as fo:
            # CIFAR-100 usa 'latin1' encoding
            data_dict = pickle.load(fo, encoding='latin1')
        print(f"File '{filename}' caricato con successo.")
        return data_dict
    except FileNotFoundError:
        print(f"Errore: File non trovato a '{file_path}'")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento o l'unpickling del file '{filename}': {e}")
        return None

# function to print a random sample of n images from a dataset
# hint: data is in dataset["data"], labels in dataset["fine_labels"] and dataset["coarse_labels"]
def print_random_samples(dataset, metadata, n=5):
    """
    Stampa un campione casuale di n immagini da un dataset CIFAR-100.

    Args:
        dataset (dict): Il dataset CIFAR-100.
        n (int): Il numero di immagini da stampare.
    """

    # Estrai le immagini e le etichette
    images = dataset["data"]
    fine_labels = dataset["fine_labels"]
    coarse_labels = dataset["coarse_labels"]

    # Seleziona n indici casuali
    indices = np.random.choice(len(images), n, replace=False)

    # Crea una figura per visualizzare le immagini
    plt.figure(figsize=(4, 20))

    for i, idx in enumerate(indices):
        # Estrai l'immagine e l'etichetta
        img = images[idx]
        fine_label = metadata["fine_label_names"][fine_labels[idx]]
        coarse_label = metadata["coarse_label_names"][coarse_labels[idx]]
        # Ridimensiona l'immagine in un array 3D (32x32x3)
        img_reshaped = img.reshape(3, 32, 32).transpose(1, 2, 0)

        # Mostra l'immagine
        plt.subplot(n, 1, i + 1)
        plt.imshow(img_reshaped)
        plt.title(f"Fine: {fine_label}\nCoarse: {coarse_label}")
        plt.axis('off')

    plt.show()