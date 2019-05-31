# Image-Restoration Team


We restore very dark images to high quality and visible images.

Here is an example from the reference paper:
![Here is an example from the reference paper](https://github.com/cchen156/Learning-to-See-in-the-Dark/blob/master/images/fig1.png)


Our purposes on this project are:

1- Reproduce the results of Learning to See in the Dark project, as can be seen here:
https://github.com/cchen156/Learning-to-See-in-the-Dark

2- Obtain results faster via optimization of the code.

3- Trying to have better results with modifications. (optional goal)

4- Testing different architectures on this problem. (optional goal)


inzva AI Projects #2 - Image Restoration Project


## Instructions

1- Paths and hyperparameters can be set at the top of test_Sony.py and train_Sony.py files.

2- The files will be read from respective input and ground truth directories.

3- The size of the deep neural network will be decided based on hyperparameters.

4- Training and test sets are generated based on the first characters of the filenames. Please refer to the code for specific implementation.

4- Output images and trained models will be saved in result and checkpoint directories.

5- For both training and test; epoch, loss, time information are printed during execution.

*Let us know if you spot any error or have any suggestions.
