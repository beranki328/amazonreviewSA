# amazonreviewSA

The Amazon Review Sentiment Anaylsis reads in a sample set of Amazon reviews on random items and returns a score from 0 to 1 based on the sentiment of the review. The program then tests itself and records the accuracy of the prediction. 

Lower scores imply a more negative review, while higher scores imply a more positive review.

  The program first reads in the dataset, found in the link below, and finds the parts of the dataset that contain the star rating and the text of the reviews. After mapping the star rating to a certain sentiment(0, 1), the reviews are stripped down to the core and tokenized. I used a Logistic Regression Algorithm for my training model, and then used the results of the test set to test the accuracy of the model. The image below shows a 94% test accuracy, making it the best performing out of Bernoulli Naive Bayes and tanh.
  
 ![image](https://user-images.githubusercontent.com/71231733/113817675-8e329680-972b-11eb-9ca0-11e4a83fc396.png)



Link to dataset: https://data.world/datafiniti/consumer-reviews-of-amazon-products
