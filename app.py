# Importing the required packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
def load_data():

    # load the dataset
    dataset = pd.read_csv("train.csv")
    X = dataset.drop(columns=['Loan_ID', 'Loan_Status'])
    y = dataset.Loan_Status

    numerical_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    #imputing missing numerical values
    imputer_num = SimpleImputer(strategy='median')
    df_num_imputed = pd.DataFrame(imputer_num.fit_transform(X.select_dtypes(include=['int64','float64'])), columns=numerical_cols)

    #imputing missing categarical values
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cat_imputed = pd.DataFrame(imputer_cat.fit_transform(X.select_dtypes(include=['object'])), columns=categorical_cols)

    cat_encoder = OneHotEncoder()
    encoded = cat_encoder.fit_transform(df_cat_imputed)
    df_cat_encoded = pd.DataFrame(encoded.toarray(),columns=cat_encoder.get_feature_names_out())
    X = pd.concat([df_num_imputed,df_cat_encoded], axis=1)

    # Take care of outliers
    X[numerical_cols] = X[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))
    # Log Transforamtion & Domain Processing
    X['LoanAmount'] = np.log(X['LoanAmount']).copy()
    X['TotalIncome'] = X['ApplicantIncome'] + X['CoapplicantIncome']
    X['TotalIncome'] = np.log(X['TotalIncome']).copy()

    # Dropping ApplicantIncome and CoapplicantIncome
    X = X.drop(columns=['ApplicantIncome','CoapplicantIncome'])

    # Train test split
    y = dataset.Loan_Status
    le = LabelEncoder()
    y = le.fit_transform(y)
    RANDOM_SEED = 6

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = RANDOM_SEED)
    return X_train, X_test, y_train, y_test


def forest(X_train, y_train):
    # RandomForest
    RANDOM_SEED = 6
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    param_grid_forest = {
        'n_estimators': [200,400, 700],
        'max_depth': [10,20,30],
        'criterion' : ["gini", "entropy"],
        'max_leaf_nodes': [50, 100]
    }

    grid_forest = GridSearchCV(
            estimator=rf,
            param_grid=param_grid_forest,
            cv=5,
            n_jobs=-1,
            scoring='accuracy',
            verbose=0
        )
    model_forest = grid_forest.fit(X_train, y_train)
    return model_forest

def lr(X_train, y_train):
    #Logistic Regression
    RANDOM_SEED = 6
    lr = LogisticRegression(random_state=RANDOM_SEED)
    param_grid_log = {
        'C': [100, 10, 1.0, 0.1, 0.01],
        'penalty': ['l1','l2'],
        'solver':['liblinear']
    }

    grid_log = GridSearchCV(
            estimator=lr,
            param_grid=param_grid_log,
            cv=5,
            n_jobs=-1,
            scoring='accuracy',
            verbose=0
        )
    model_log = grid_log.fit(X_train, y_train)
    return model_log

#Decision Tree
def tree(X_train, y_train):
    RANDOM_SEED = 6
    dt = DecisionTreeClassifier(
        random_state=RANDOM_SEED
    )

    param_grid_tree = {
        "max_depth": [3, 5, 7, 9, 11, 13],
        'criterion' : ["gini", "entropy"],
    }

    grid_tree = GridSearchCV(
            estimator=dt,
            param_grid=param_grid_tree,
            cv=5,
            n_jobs=-1,
            scoring='accuracy',
            verbose=0
        )
    model_tree = grid_tree.fit(X_train, y_train)
    return model_tree



mlflow.set_experiment("Loan_prediction")

# Model evelaution metrics
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1, auc)


def mlflow_logging(model, X, y, name):

     with mlflow.start_run() as run:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        #mlflow.set_tag("run_id", run_id)
        pred = model.predict(X)
        #metrics
        (accuracy, f1, auc) = eval_metrics(y, pred)
        # Logging best parameters from gridsearch
        mlflow.log_params(model.best_params_)
        #log the metrics
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # Logging artifacts and model
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)

        mlflow.end_run()
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model_tree = tree(X_train,y_train)
    model_log = lr(X_train,y_train)
    model_forest = forest(X_train,y_train)
    mlflow_logging(model_tree, X_test, y_test, "DecisionTreeClassifier")
    mlflow_logging(model_log, X_test, y_test, "LogisticRegression")
    mlflow_logging(model_forest, X_test, y_test, "RandomForestClassifier")
