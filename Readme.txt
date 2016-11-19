Shantanu Thakoor, 13D100003
Karan Vaidya, 130050019
Sai Sandeep, 130050052
Shudhatma Jain, 130050024

python makeData.py generates the numpy array which is used as training data, from the files present in "train" folder (similarly validation data from "valid" folder). Hence we generate train_data.npy and data.npy

python try_nn.py 100 30 0.001
This will execute the tensorflow code; making a neural network of hidden layers size 100 and 30, and learning rate 0.001.
Alongside learning, at some periodic intervals it will also print the accuracy and true/false positive/negative rates on the test data.

python model.py
This will execute the adaboost algorithm, and at the end will give the accuracy on the test data.



Emotion Classification : 
python convolutional.py
Executes CNN code for emotion detection in images. Contains code for both binary and multiple emotion classification.