# %%
from functions import get_data_dir, load_cifar_datafile, print_random_samples
data_dir = get_data_dir(subfolder_name="cifar_100")
# %%
train_data = load_cifar_datafile(data_dir, 'train')
meta_data = load_cifar_datafile(data_dir, 'meta')
# %%
print_random_samples(train_data, meta_data, n=5)
# %%