# Phishing Email Classification Using Transformers

This project involves training a transformer model to classify phishing emails using a dataset from [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails).

## Dataset

The dataset used for this project contains two types of emails: "Safe Email" and "Phishing Email." In total, the dataset consists of 11,322 safe emails and 7,328 phishing emails. 

The dataset was split into training, validation, and testing sets. The text of the emails was tokenized using a BERT tokenizer (`bert-base-uncased`).

## Approach

A transformer model (also `bert-base-uncased`) was fine-tuned to classify the emails into "Safe" and "Phishing" emails. The model was trained using the AdamW optimizer, with a learning rate of 5e-5 for three epochs.

## Results

The trained model achieved a very high accuracy of ~98.1% on the validation set. This suggests that the model is effectively learning to distinguish between safe and phishing emails.

In terms of error analysis, a few instances of emails were printed out with their predicted and actual labels, which the model had predicted wrong. This helped understand what the model might be missing.

The model can be used to predict unseen user text for phishing attempts.

## Future Improvements

While the current model performs well on the validation set, here are a few steps that can be taken to further improve the model's outcomes:

- **Incorporate More Data**: Given more computational resources, the model could potentially be trained on a larger dataset. This would allow the model to learn from a wider variety of email patterns, likely improving its phishing detection capabilities.

- **Use Other Metrics**: In addition to accuracy, other metrics such as precision, recall, F1 score, or AUC-ROC can be used to evaluate the model from different aspects, especially in the case of imbalanced classes.

- **Hyperparameter Tuning**: Different hyperparameters such as learning rate or number of training epochs could be tested to see if they improve model performance.

- **Use Updated Models**: As NLP research progresses, newer and potentially better-performing models are being released. Keeping up-to-date with the latest models and experimenting with them could lead to performance improvements. 

- **Model Interpretability**: Techniques for making the model's predictions interpretable can be applied, which can help in understanding why the model is making certain predictions.
