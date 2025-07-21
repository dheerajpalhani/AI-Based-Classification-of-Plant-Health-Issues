
This project is a Tkinter-based desktop application that leverages machine learning algorithms to detect and classify plant diseases using agricultural datasets. The goal is to provide farmers or agricultural professionals with a decision support tool that recommends the appropriate pesticide based on disease diagnosis.

🔍 Features
GUI Interface with role-based login for Admins and Users

Dataset Upload & Visualization: View and explore dataset with various plots like bar plots, violin plots, scatter plots, and heatmaps

Preprocessing & Label Encoding: Automatically handles missing values and encodes categorical data

Class Imbalance Handling: Uses SMOTE to balance the dataset

Train-Test Split: Splits the dataset into training and testing subsets (80/20)

ML Model Integration:

Naive Bayes

SVM

KNN (Proposed Classifier)

Decision Tree (Proposed Classifier)

Model Persistence using joblib for saving and reloading models

Performance Evaluation with metrics such as accuracy, precision, recall, and F1-score

Confusion Matrix Visualization

Pesticide Suggestion: Based on predicted disease class

Login/Signup System connected to MySQL for role-based access

🛠️ Technologies Used
Python

Tkinter for GUI

pandas, numpy, seaborn, matplotlib for data processing and visualization

scikit-learn for ML models

imblearn (SMOTE) for handling imbalanced datasets

MySQL for user login and role management

💡 How to Use
Admin/Users can register and login

Admins can:

Upload and preprocess a dataset

Run EDA (Exploratory Data Analysis)

Apply SMOTE

Train multiple classifiers

View performance metrics and confusion matrices

Users can:

Upload test data

Predict disease type

Get suggested pesticide based on prediction

📂 Dataset Requirement
The dataset should include agricultural/plant-related features with a column named Label representing the plant disease class.

📈 Future Improvements
Add comparison graphs between model performances

Enable CSV download of prediction results

Support for image-based disease detection using CNNs

