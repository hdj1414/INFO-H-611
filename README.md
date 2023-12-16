# Naive Bayes Classification
Utilization of Naive Bayes Classification in Data Science - Explaining based on using NMFS-AFSC Longline Sablefish Survey of Alaska
#### *- Mathematical and Logical Foundations of Informatics - INFO-H 611 (26951) Fall 2023*
#### *- Written By - Hasaranga Jayathilake (hdjayath@iu.edu)*

# What is Naive Bayes Classification

## Introduction
If you want to learn how to classify data into different categories, you might want to try Naive Bayes. It is a simple, fast, and flexible algorithm that can handle many kinds of data, from emails to reviews. In this article, we'll explain how Naive Bayes works, and what it can and can't do.

## Understanding Naive Bayes
### The Algorithm
Naive Bayes is based on a formula called Bayes' theorem, a basic rule of probability. It helps us calculate how likely something is based on what we already know. For example, if we know that someone likes pizza, we can use Bayes' theorem to estimate how likely they are to order pizza for dinner. Naive Bayes uses this formula to assign data points to classes based on their features. For example, if we have a dataset of emails and their features, such as sender, subject, and words, we can use Naive Bayes to predict whether an email is spam or not.

### Strengths
- **Simplicity:** Naive Bayes is easy to learn and use. You don't need a lot of math or programming skills to apply it.

- **Efficiency:** Naive Bayes is very fast and can handle large amounts of data without slowing down.

- **Text Classification:** Naive Bayes is especially good at dealing with text data, such as natural language. It can help us analyze the emotions and opinions of people, or filter out unwanted messages.

### Weaknesses
- **Assumption of Independence:** The algorithm assumes that the data features are independent, which means they don't affect each other. This might not be true in reality. For example, the words in an email might depend on the sender or the subject.

- **Limited Expressiveness:** The algorithm might not be able to capture complex patterns or relationships in the data because it is too simple. It might miss some important details or nuances that other algorithms can find.

## Simplified Process of the Naive Bayes Algorithm.

### **Pseudocode for Naive Bayes Classification**

1 Step 1: Preprocess the data
 - Tokenize and clean the text (for text classification)
 - Handle missing values and outliers

2 Step 2: Split the dataset
 - Divide the data into training and testing sets

3 Step 3: Train the Naive Bayes model
 - Calculate class probabilities
 - Estimate feature probabilities given each class

4 Step 4: Make predictions
 - For each new data point:
   - Calculate the likelihood of features given each class
   - Multiply with class probabilities
   - Choose the class with the highest probability as the prediction
  
## Adaptability to Data Types
Working with Different Data Naive Bayes can handle many kinds of data. For example, in text classification, it can tell how people feel from the words they use. Its code is flexible and can be changed slightly for different data features

Among the many ways to classify data, Naive Bayes is easy but powerful. Learning how it works and using it wisely can help data scientists solve many problems. Naive Bayes is more than just a method, it is proof that simplicity can lead to great results in the world of data science.

> As an example, using fishing data from ***NMFS-AFSC Longline Sablefish Survey from Alaska***, Naive Bayes classification can be implemented in the below manner.

### Pseudocode for Naive Bayes Classification in the Fishing Industry

1 Step 1: Preprocess the fishing data
 - Handle missing values related to catch frequencies
 - Categorize features like fishing methods and environmental conditions

2 Step 2: Split the dataset
 - Divide the data into training and testing sets, ensuring the representation of diverse conditions

3 Step 3: Train the Naive Bayes model
 - Calculate probabilities of catching specific fish types given different features
 - Estimate class probabilities for different catch frequencies

4 Step 4: Make predictions for new data
 - For each set of environmental conditions and fishing methods:
   - Calculate the likelihood of catching each fish type
   - Multiply with class probabilities
   - Predict the most probable catch frequency

> Below code block is a walk through of the code block provided in the "Code" file of this repository and explain each step, discussing how Naive Bayes classification works in the context of predicting catch frequency.



# Code Explanation:

## Technology that used in this code:
- Google Colab
- Pyspark
- Libraries from Python which are relevant with the pyspark

```
# Define features
feature_columns = ["distance_fished", "hachi", "year", "soak_time", "starting_depth",
                   "ending_depth", "surface_temperature", "catch_freq"]
```
> The code starts by defining the feature columns, which are the attributes used for prediction. These features include parameters such as distance_fished, hachi, year, soak_time, starting_depth, ending_depth, surface_temperature, and catch_freq.

```
# Create a Vector Assembler
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
```
> The VectorAssembler is a feature transformer that combines a given list of columns into a single vector column. In this case, it combines the specified feature columns into a new column named "features."

```
# Define Naive Bayes classifier
nb = NaiveBayes(labelCol="catch_freq", featuresCol="features")
```
> The NaiveBayes class is the implementation of the Naive Bayes algorithm in PySpark's machine learning library. It is set up to predict the "catch_freq" column based on the features assembled in the "features" column.

```
# Create a pipeline
pipeline = Pipeline(stages=[assembler, nb])
```
> A pipeline is created to organize and chain multiple stages of the machine learning process. In this case, the pipeline consists of the VectorAssembler and NaiveBayes stages.

```
# Split the data into training and testing sets
(training_data, testing_data) = df.randomSplit([0.8, 0.2], seed=1234)
```
> The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing. The seed parameter ensures reproducibility.

```
# Define the label columns and combine them into a single "label" column in the testing data
label_columns = ["distance_fished", "hachi", "year", "soak_time", "starting_depth", "ending_depth", "surface_temperature", "catch_freq"]
testing_data = testing_data.withColumn("label", col(label_columns[0]).cast("double"))
for label_col in label_columns[1:]:
    testing_data = testing_data.withColumn("label", testing_data["label"] + col(label_col).cast("double"))
```
> The label columns are specified, and a new column "label" is created in the testing data by combining the numerical values of the label columns.

```
# Make predictions on the testing set
predictions = model.transform(testing_data)
```
> The model is used to make predictions on the testing set.

```
# Evaluate the model using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
```
> The accuracy of the model is evaluated using the MulticlassClassificationEvaluator, comparing the predicted values with the actual labels.

```
testing_data.select("label").distinct().show()
```
> This line displays the distinct values of the label column in the testing data.

```
predictions.select("prediction").distinct().show()
```
> This line displays the distinct predicted values in the "prediction" column.

## Conclusion:
In summary, the provided code block demonstrates the application of Naive Bayes classification to predict catch frequency based on various features. The model is trained on the training set, and predictions are made on the testing set, followed by an evaluation of accuracy. Understanding each step is crucial for effectively applying machine learning techniques to real-world datasets. Naive Bayes, known for its simplicity and efficiency, proves to be a valuable tool in predicting catch frequency in this context.







