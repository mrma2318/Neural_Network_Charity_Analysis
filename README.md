# Neural_Network_Charity_Analysis
## Overview: Assist Beks in creating a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. 
### Purpose: The purpose of this project is to using machine learning and neural networks to assist in predicting whether applicants will be successful.

## Analysis
- For this project, using my knowledge of Pandas and Scikit-Learn, I preprocessed the dataset in order to compile, train, and evaulate the neural network model. When preprocessing the data, the EIN and NAME variables, were considered my target variables, while the remaining variables were my feature variables for my model. 

- Once I dropped my target variables I could then determine the number of unique values for each column. Those with more than 10 unique values, I then determined the number of data points for each unique value. Next, I created a density plot to determine the distribution of the column values, and used the plot to create a cutoff point to bin "rare" categorical variables in a new column. 

- I then encoded the categorical variables using the one-hot encoding, and placed the variables in a new DataFrame. Next, I merged the one-hot encoding DataFrame with the original DataFrame, and droped the originals, Image 1.

### Image 1: Application DataFrame
![Application DataFrame](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/2eb848a5a6b75b7e4748c31a0c07d07163bec015/images/Image1.png)

- Then, I split the preprocessed data into features and target arrays, as well as into training and testing datasets. Next, I standardized the numberical variables using Scikit-Learn's StardardScaler class, and scaled the data. 

- Next, I used my knowledge of TensorFlow to design a neural network to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. I created a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras. Then I created the first hidden layer and choose an appropriate activation function. For this model, I used relu for the first hidden layer. I also added a second hidden layer, and used relu for the activation function as well. Lastly, I created an output layer with an activation function sigmoid and checked the structure of the model, Image 2.

### Image 2: Nerual Network Model
![Neural Network Model](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/2eb848a5a6b75b7e4748c31a0c07d07163bec015/images/Image2.png)

- Now I can compile and train the model. Afterwards, I created a callback that saves the model's weights every 5 epochs and evaluated the model using the test data to determine the loss and accuracy, Image 3. 

### Image 3: Loss and Accuracy Scores
![Loss and Accuracy Scores](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/2eb848a5a6b75b7e4748c31a0c07d07163bec015/images/Image3.png)

- Since I want an accuracy score higher than 75%, I optimized my model in order to achieve a higher percentage. I adjusted the input data to ensure that there were no variables or outliers that causing confusion in the model. To do this, I tried a variety of things including dropping more or fewer columns, creating more bins for rare occurrences in columns, and increasing or decreasing the number of values for each bin. I also tried adding more neurons to a hidden layer, adding more hidden layers, using different activation functions for the hidden layers, and adding the number of epochs to the training regimen. 

## Results
### Data Preprocessing
- 

### Compiling, Training, and Evaluating the Model

## Summary