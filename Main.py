from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import tkinter as tk
from tkinter import messagebox
import pymysql

import matplotlib.pyplot as plt
from tkinter import simpledialog
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

labels = ['diaporthe-stem-canker', 'charcoal-rot', 'rhizoctonia-root-rot',
       'phytophthora-rot', 'brown-stem-rot', 'powdery-mildew',
       'downy-mildew', 'brown-spot', 'bacterial-blight',
       'bacterial-pustule', 'purple-seed-stain', 'anthracnose',
       'phyllosticta-leaf-spot', 'alternarialeaf-spot',
       'frog-eye-leaf-spot', 'diaporthe-pod-&-stem-blight',
       'cyst-nematode', 'herbicide-injury']

pesticides = ['Tebuconazole', 'Azoxystrobin', 'Carboxin',
       'Mefenoxam', 'Crop Rotation', 'Triazoles',
       'Metalaxyl', 'Tebuconazole', 'Copper Oxychloride',
       'Oxytetracycline', 'Azoxystrobin', 'Difenoconazole',
       'Thiophanate-methyl', 'Azoxystrobin',
       'Difenoconazole', 'Propiconazole',
       'Oxamyl', 'Amino Acid & Seaweed Extracts']

def uploadDataset():
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))

def bar_plot(df, column):
    """Bar plot showing mean values of a numerical column per Label."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Label", y=column, data=df, estimator=lambda x: x.mean(), ci=None)
    plt.title(f'Bar Plot of {column} (Mean) by Label')
    plt.xlabel("Label")
    plt.ylabel(f'Mean {column}')
    plt.show()

def violin_plot(df, column):
    """Violin plot for a numerical column grouped by Label."""
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Label", y=column, data=df)
    plt.title(f'Violin Plot of {column} by Label')
    plt.xlabel("Label")
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.show()

def strip_plot(df, column):
    """Strip plot for a numerical column grouped by Label."""
    plt.figure(figsize=(8, 5))
    sns.stripplot(x="Label", y=column, data=df, jitter=True, alpha=0.5)
    plt.title(f'Strip Plot of {column} by Label')
    plt.xlabel("Label")
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.show()

def histogram(df, column):
    """Histogram for a numerical column, colored by Label."""
    plt.figure(figsize=(8, 5))
    for label in df["Label"].unique():
        subset = df[df["Label"] == label]
        plt.hist(subset[column], bins=30, alpha=0.5, label=str(label))
    plt.title(f'Histogram of {column} by Label')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend(title="Label")
    plt.show()

def scatter_plot(df, x_column, y_column):
    """Scatter plot for two numerical columns, colored by Label."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x_column, y=y_column, hue="Label", data=df, alpha=0.6)
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend(title="Label")
    plt.show()

def strip_plot(df, column):
    """Strip plot for a numerical column grouped by Label."""
    plt.figure(figsize=(8, 5))
    sns.stripplot(x="Label", y=column, data=df, jitter=True, alpha=0.5)
    plt.title(f'Strip Plot of {column} by Label')
    plt.xlabel("Label")
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.show()

def correlation_heatmap(df):
    """Heatmap of correlation matrix for numerical columns."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
    
def EDA():
    global filename, dataset, labels
    df=dataset
    bar_plot(df, "temp")                  # 1. Bar plot
    violin_plot(df, "leafspot-size")       # 2. Violin plot
    histogram(df, "plant-growth")          # 4. Histogram
    scatter_plot(df, "temp", "germination")# 5. Scatter plot
    strip_plot(df, "stem")                 # 6. Strip plot
    correlation_heatmap(df)                # 7. Correlation heatmap
    scatter_plot(df, "leafspot-size", "germination")# 5. Scatter plot

def DatasetPreprocessing():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder

    #dataset contains non-numeric values but ML algorithms accept only numeric values so by applying Lable
    #encoding class converting all non-numeric data into numeric data
    dataset.fillna(0, inplace = True)
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            label_encoder.append(le)    
    
    X = dataset.drop(['Label'], axis = 1)
    Y = dataset['Label']
    labels, label_count = np.unique(dataset['Label'], return_counts=True)

    text.insert(END,"Dataset Normalization & Preprocessing Task Completed\n\n")
    text.insert(END,str(dataset)+"\n\n")

def Dataset_SMOTE():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder
    labels, label_count = np.unique(dataset['Label'], return_counts=True)
    #dataset preprocessing such as replacing missing values, normalization and splitting dataset into train and test
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(X, Y)
# Count after SMOTE
    labels_resampled, label_count_resampled = np.unique(Y, return_counts=True)
# Plotting
    plt.figure(figsize=(10, 5))
# Before SMOTE
    plt.subplot(1, 2, 1)
    plt.bar(labels, label_count, color='skyblue', alpha=0.8)
    plt.xlabel("Output Type")
    plt.ylabel("Count")
    plt.title("Before SMOTE")

    # After SMOTE
    plt.subplot(1, 2, 2)
    plt.bar(labels_resampled, label_count_resampled, color='lightgreen', alpha=0.8)
    plt.xlabel("Output Type")
    plt.ylabel("Count")
    plt.title("After SMOTE")
    plt.tight_layout()
    plt.show()


def Train_test_splitting():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder
    global X_train, X_test, y_train, y_test, scaler

 
    #splitting dataset into train and test where application using 80% dataset for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total Data found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for testing   : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

#now train existing algorithm    
def Existing_Classifier():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []
    
    if os.path.exists('model/naive_bayes_model.pkl'):
        classifier = joblib.load('model/naive_bayes_model.pkl')
    else:                       
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, 'model/naive_bayes_model.pkl')

    y_pred_bnb = classifier.predict(X_test)
    calculateMetrics("Existing Gaussian NBC", y_test, y_pred_bnb)
    
def Existing_Classifier1():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []
    
    if os.path.exists('model/SVC.pkl'):
        classifier = joblib.load('model/SVC.pkl')
    else:                       
        classifier = SVC()
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, 'model/SVC.pkl')

    y_pred_bnb = classifier.predict(X_test)
    calculateMetrics("SVM", y_test, y_pred_bnb)

def Proposed_Classifier():
    global classifier
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test
    if os.path.exists('model/KNeighborsClassifier.pkl'):
        # Load the model from the pkl file
        classifier = joblib.load('model/KNeighborsClassifier.pkl')
    else:
        # Train the classifier on the training data
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train)
        # Save the model weights to a pkl file
        joblib.dump(classifier, 'model/KNeighborsClassifier.pkl')
    
    y_pred = classifier.predict(X_test)
    calculateMetrics("Existing KNN", y_test, y_pred)
    
def Proposed_Classifier1():
    global classifier
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test
    if os.path.exists('model/DecisionTreeClassifier.pkl'):
        # Load the model from the pkl file
        classifier = joblib.load('model/DecisionTreeClassifier.pkl')
    else:
        # Train the classifier on the training data
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        # Save the model weights to a pkl file
        joblib.dump(classifier, 'model/DecisionTreeClassifier.pkl')
    
    y_pred = classifier.predict(X_test)
    calculateMetrics("Proposed Decision Tree", y_test, y_pred)

     
def predict():
 
    global classifier,pesticides
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")#upload test data
    dataset = pd.read_csv(filename)#read data from uploaded file
    dataset.fillna(0, inplace = True)#removing missing values
    predict = classifier.predict(dataset)

    for i in range(len(X)):
        text.insert(END,"Sample Test Data:" +str(dataset.iloc[i]))
        text.insert(END,"Output Classified As ===> "+labels[int(predict[i])]+"\n")
        text.insert(END,"Pesticides Suggested As ===> "+pesticides[int(predict[i])]+"\n")

        text.insert(END,"\n\n\n")



def connect_db():
    return pymysql.connect(host='localhost', user='root', password='root', database='sparse_db')

# Signup Functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
                cursor.execute(query, (username, password, role))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login Functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s"
                cursor.execute(query, (username, password, role))
                result = cursor.fetchone()
                conn.close()
                if result:
                    messagebox.showinfo("Success", f"{role} Login Successful!")
                    login_window.destroy()
                    if role == "Admin":
                        show_admin_buttons()
                    elif role == "User":
                        show_user_buttons()
                else:
                    messagebox.showerror("Error", "Invalid Credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

# Admin Button Functions
def show_admin_buttons():
    clear_buttons()
    tk.Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=70, y=200)
    tk.Button(main, text="EDA", command=EDA, font=font1).place(x=230, y=200)
    tk.Button(main, text="Preprocessing", command=DatasetPreprocessing, font=font1).place(x=300, y=200)
    tk.Button(main, text="SMOTE", command=Dataset_SMOTE, font=font1).place(x=500, y=200)
    tk.Button(main, text="Train Test Splitting", command=Train_test_splitting, font=font1).place(x=650, y=200)
    tk.Button(main, text="NBC", command=Existing_Classifier, font=font1).place(x=900, y=200)
    tk.Button(main, text="SVM", command=Existing_Classifier1, font=font1).place(x=1000, y=200)
    tk.Button(main, text="KNN", command=Proposed_Classifier, font=font1).place(x=1100, y=200)
    tk.Button(main, text="Decision Tree", command=Proposed_Classifier1, font=font1).place(x=1200, y=200)

# User Button Functions
def show_user_buttons():
    clear_buttons()
    tk.Button(main, text="Prediction", command=predict, font=font1).place(x=550, y=200)
    #tk.Button(main, text="Comparison Graph", command=comparison_graph, font=font1).place(x=400, y=400)

# Clear buttons before adding new ones
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()
    

# Main tkinter window
main = tk.Tk()
main.geometry("1300x1200")
main.title("AI based Classification on Plant Health Issues For Agriculture Optimization")

# Title
font = ('times', 18, 'bold')
title = tk.Label(main, text="AI based Classification on Plant Health Issues For Agriculture Optimization", bg='white', fg='black', font=font, height=2, width=100)
title.pack()

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=170)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=250)
text.config(font=font1)


# Admin and User Buttons
font1 = ('times', 14, 'bold')


tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=2, bg='LightBlue').place(x=100, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=2, bg='LightGreen').place(x=400, y=100)


admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=2, bg='LightBlue')
admin_button.place(x=700, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=2, bg='LightGreen')
user_button.place(x=1000, y=100)

main.config(bg='OliveDrab2')
main.mainloop()