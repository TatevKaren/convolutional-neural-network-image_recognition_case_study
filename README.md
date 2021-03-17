# dog_cat_image_recognition_cnn
Image recognition model: Computer Vision case study to classify an image to a binary class. Image recognition model based on Convolutional Neural Network (CNN) to identify from an image whether it is dog or a cat image. Classifying Dog and Cat Images. In this case study we use 8000 images of dogs and cats to train and test a Convolutional Neural Network (CNN) model that takes an image as an input and give as an output a class of 0 (cat) or 1 (dog) suggesting whether it is a dog or a cat picture. This image recognition model is based on CNN. 
The process includes the following steps and is captured in this <a href ="https://github.com/TatevKaren/computer-vision-case-study/blob/main/Convolutional_Neural_Network_Case_Study.py" >Python Code</a>:
- Data Preprocessing
- Buidling CNN Initialization 
- Training and Testing CNN
- Model validation

<br><br>

## Image examples used to train the CNN
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/dog.31.jpg?raw=true"
  width="210" height="200">
   <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat.3.jpg?raw=true"
  width="210" height="200">
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/dog.24.jpg?raw=true"
  width="210" height="200">   
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat.12.jpg?raw=true"
  width="210" height="200">    
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat.1.jpg?raw=true"
  width="210" height="200">
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/dog.4.jpg?raw=true"
  width="210" height="200">    
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat.8.jpg?raw=true"
  width="210" height="200">
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/dog.45.jpg?raw=true"
  width="210" height="200">  
</p>
<br><br>


## Convolutional Neural Networks (CNN)
This case study is based on CNN model and the <a href="https://github.com/TatevKaren/computer-vision-case-study/blob/main/Convolutional_Neural_Networks_Case_Study.pdf">CASE STUDY PAPER</a> includes detailed description of all the steps and processes that CNN's include such as:
- Convolutional Operation
- Pooling
- Flattening
<p>
    <img src="https://github.com/TatevKaren/computer-vision-case-study/blob/main/cnn_summary.png?raw=true"
  width="700" height="270">
</p>
<br><br><br>

## Model Evaluation
Important evaluation steps, described in detail in <a href="https://github.com/TatevKaren/computer-vision-case-study/blob/main/Convolutional_Neural_Networks_Case_Study.pdf"> CASE STUDY PAPER </a> , that help the CNN model to train and make accurate predictions such as:
- Loss Functions for CNN (SoftMax and Cross-Entropy)
- Loss Function Optimizers (SGD and Adam Optimizer)
- Activation Functions (Rectifier and Sigmoid)
<br><br><br>

## Deep Learning libraries: Tensorflow & Keras 
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras ImageDataGenerator Library.png?raw=true"
  width="500" height="330">
</p>
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras ImageDataGenerator Library2.png?raw=true"
  width="600" height="330">
</p>
<p>
   <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras load_img function.png?raw=true"
  width="600" height="90">
   <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras load_to_array function.png?raw=true"
  width="600" height="90">
</p>
<br>
Source: https://keras.io/api/preprocessing/image/#imagedatagenerator-class

<br><br>

## Prediction results snapshot
Pair of images used to evaluate the trained and tested CNN model to observe to which class does the model classify the following pictures: Dog or Cat
<br>
### Input Image 1
<p align="left">
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat_or_dog_1.jpg?raw=true"
  width="300" height="200">

</p>
<br>

### Input Image 2
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat_or_dog_2.jpg?raw=true"
  width="300" height="200">
</p>
<br><br>
After compiling the model thye modelly accurately classifies the first picture to a dog class and the second picture to a cat class. Following is a snapshot of a Python output.
<br><br>
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Prediction_Snapshot.png?raw=true"
  width="900" height="600">
</p>


