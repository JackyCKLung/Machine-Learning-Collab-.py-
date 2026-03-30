import numpy as np

# Load the data
train_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\train.npz')
val_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\val.npz')
test_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\test.npz')

# Inspect the contents
print(train_data.files)  # This will list the arrays stored in the file

