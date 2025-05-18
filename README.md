# CIFAR-10 Image Classification using CNN

Built and trained a Convolutional Neural Network to classify images from the CIFAR-10 dataset.
Final model achieved a training accuracy of 89.79% and a test accuracy of 89.72%

## Project Structure
- 'data/' - stores CIFAR10 dataset.
- 'experiments' - contains saved config, model and performance graphs from current experiment.
- 'src/'
  - 'scripts/' - contains training and validation scripts that run per epoch.
  - 'utils/' - contains data loading and augmentation, experiment logging and graphing scripts.
  - 'model.py' - contains model code.
  - 'config.py' - configuration file to define experiment and model parameters.
  - 'test.py', 'train.py' - set up and run training and testing.
- 'environment.yml' - Conda environment config file.

## Requirements
- Conda 25.3.1 (Miniconda specifically was used for this project).
- Python 3.11
- Further package requirements are explicitly stated in the environemnt.yml file.

## Setup Instructions
- Once in the unzipped folder directory, open a terminal and run:

  - conda env create -f environemnt.yml
  - conda activate imlocoursework

## Running the project
- A config.py file in src contains all parameters to recreate the training and testing of the provided model.

### Training 
- The model was trained using a set random seed of 22, to train without a set seed, simply replace the number with 'None'.
- If you wish to retrain the model multiple times, make sure to change the experiment name in config.py as to
  not overwrite previously trained models along with their logs and performance graphs.
- To train the model, in the project root directory, run:
  
  conda run -n imlocoursework python -m src.train
  
- This will create a data folder in the root directory, and once training is complete, will save the model config,
  performance graphs and saved model in the experiments folder.
- As the model is being trained, test and validation losses are printed in the terminal per epoch.

### Testing
- To test the model, in the project root directory, run:

  conda run -n imlocoursework python -m src.test

- By default, this will test the submitted trained model, with its path also defined in config.py.
- Test loss and test accuracy are displayed in the terminal after testing.
