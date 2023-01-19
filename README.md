# Neural_Network_Charity_Analysis
## Overview: Assist Beks in creating a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. 
### Purpose: The purpose of this project is to using machine learning and neural networks to assist in predicting whether applicants will be successful.

## Analysis
The analysis script can be found by going to [AlphabetSoupCharity.h5](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/ad8763d676cf1537b8e0eef4b14f64faa508ff00/AlphabetSoupCharity.h5) and [AlphabetSoupCharity.ipynb](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/ad8763d676cf1537b8e0eef4b14f64faa508ff00/AlphabetSoupCharity.ipynb).

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
- In the initial preprocessing of the data, the target variables for my model were the "EIN" and "NAME" variables. However, when optimizing the data to try and reach 75% accuracy, "USE_CASE" was also one of my target variables. All other variables were my feature variables for my model. 

### Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take to try and increase model performance?

- In the inital preprocessing of the data, I used a total of 110 neurons, broken up in two hidden layers. The first hidden layer had 80 neurons with a relu activation function. While the second hidden layer had 30 neurons with a relu activation function as well. Lastely, I had an output layer that used the sigmoid activation function. The reason I used the relu activation function is because the relu function is faster to compute compared to the sigmoid. 

- However, I was not able to achieve target model performance with an accuracy of 73%. Therefore, I optimized my model to try and achieve a higher accuracy of 75% or greater. I tried to optimize my model at least three times. The first attempt, is when I used "USE_CASE" as a target variable for my model and changed the thresholds for binning. The accuracy score however, was still 73%, Image 4. 

The script for the optimization code can be found by going to [AlphabetSoupCharity_Optimization.ipynb](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/ad8763d676cf1537b8e0eef4b14f64faa508ff00/AlphabetSoupCharity_Optimization.ipynb).

### Image 4: Optimization 1 Accuracy
![Optimization 1 Accuracy](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/0c47da29d69b030bbbd7b96e9331fa160a0f6d28/images/Image4.png)

- For the second attempt, I added a second hidden layer and increased the total number of neurons from 110 to 140. In the first hidden layer I changed it from 80 to 100, in the second layer, I kept the total number of neurons 30, and in the third hidden layer I added, I had 10 neurons. However, I still ended up with an accuracy of 73%, Image 5.

### Image 5: Optimization 2 Accuracy
![Optimization 2 Accuacy](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/0c47da29d69b030bbbd7b96e9331fa160a0f6d28/images/Image5.png)

- Lastly, for the third attempt I used different activation functions for the hidden layers and added epochs to the training regimen. For the activation functions on all hidden layers to the sigmoid activation function. Then I changed the epochs from 100 to 200. When I ran the accuracy test though, I still had a 73% accuracy, Image 6. 

### Image 6: Optimization 3 Accuracy
![Opitimization 3 Accuracy](https://github.com/mrma2318/Neural_Network_Charity_Analysis/blob/0c47da29d69b030bbbd7b96e9331fa160a0f6d28/images/image6.png)

## Summary
- In summary, I was not able to achieve an accuracy of 75% or higher for the model. Therefore, this is not the best model capable of predicting whether applicants will be succeessful if funded by Alphabet Soup. I would recommend using one of the Learning Classifier Models such as the Random Forest Classifier to try and see if they could achieve an accuracy of 75% or greater. The learning classifier algorithms also take less time to train. In addition, the Random Forest Classifier can hangle larger datasets efficiently, and provide a higher level of accuracy in predicting outcomes compared to the decision tree algorithms. 