import mlflow
import dagshub
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("https://dagshub.com/suyashverma0/mlops-exp-with-mlflow.mlflow")


dagshub.init(repo_owner='suyashverma0', repo_name='mlops-exp-with-mlflow', mlflow=True)






# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

max_depth = 14
n_estimators =20
mlflow.set_experiment("mlops-exp-2")
mlflow.autolog()
with mlflow.start_run() as run:
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)


   

    # creating a confusion matrix plot
    cm= confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel('prediction')
    plt.title('confusion matrix')

    plt.savefig('confusion_matrix.png')


    
    mlflow.log_artifact(__file__)
    mlflow.set_tags({'author':'suyash','project':'wine_project'})

    






print(accuracy)

