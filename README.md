To create a new NN model and save it run:
```
python3 train_nn.py
```
This also saves the Keras model as ```facial_recognition_model<DATE CREATED>.h5```


To run the saved model against your test cases run:
```
python3 run_model <PATH_TO_MODEL.h5> <IMAGES_DATA_FILE.mat>
```
This also saves the pickled predicted labels array to ```predicted_labels.p```

The ```requirements.txt``` lists the python3 dependencies for the project.
Run ```pip3 install requirements.txt``` incase of missing dependencies.
