1. What is AI/ML?
AI (Artificial Intelligence): Teaching computers to think and act like humans.
Machine Learning (ML): A subset of AI where computers learn patterns directly from data instead of being explicitly programmed.

2. Learning Types
Supervised Learning: Learning with labeled data (inputs + correct answers). Example: Predicting house prices or detecting spam.
Unsupervised Learning: Finding hidden patterns in unlabeled data. Example: Grouping similar customers by shopping habits.

3. What is a Model & How Training Works
Model: A math formula or algorithm that turns inputs into predictions.
Training: Giving data to the model so it can adjust its internal rules and improve prediction accuracy.

Dataset Split:
Train Set (~70%): Learns patterns.
Validation Set (~15%): Fine-tunes settings.
Test Set (~15%): Final exam on unseen data.

4. Data Cleaning (Preprocessing)
Bad data leads to bad predictions ("garbage in, garbage out").
Missing Data: Fill in blanks using average values (imputation) or drop incomplete rows.
Outliers: Catch extreme values that skew results and cap or remove them.
Categorical Encoding: Turn text categories (like "Red", "Blue") into numbers that math models can process.
Feature Scaling: Rescale different numbers (e.g., age vs. salary) to a shared scale so large numbers don't overpower smaller ones.

5. Overfitting vs. Underfitting
Underfitting: Model is too simple; fails on both training and new data.
Overfitting: Model memorizes training data too closely (including noise); fails on new data.

6. Key Evaluation Metrics
Accuracy: Percentage of total correct predictions.
Precision: Out of everything predicted positive, how many were actually positive? (Avoids false alarms).
Recall: Out of all actual positive cases, how many did the model catch? (Avoids missing critical cases).

F1-Score: A balanced score combining Precision and Recall.
