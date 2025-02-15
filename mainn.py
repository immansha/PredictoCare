import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

def create_model(data):
    # Splitting features and target variable
    X = data.drop('diagnosis', axis=1)  # Fixing the drop syntax
    Y = data['diagnosis']

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    #test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:',accuracy_score(Y_test,y_pred))
    print("Classification report:\n", classification_report(Y_test,y_pred))

    return model, scaler

def get_clean_data():
    data = pd.read_csv("C:\\Users\\hp\\Downloads\\breast cancer proj\\data.csv")

    data = data.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data  

def main():
    data = get_clean_data() 
    
    model, scaler = create_model(data)
    
    with open('C:\\Users\\hp\\Downloads\\breast cancer proj\\model.pkl','wb') as f:
        pickle.dump(model, f)
    with open('C:\\Users\\hp\\Downloads\\breast cancer proj\\scaler.pkl','wb') as f:
        pickle.dump(scaler, f)



if __name__ == "__main__":
    main()
