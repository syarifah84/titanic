import pandas as pd
import streamlit as st


st.title('CLASSIFICATION')

st.sidebar.write("""
A classification demo using [titanic](https://www.kaggle.com/c/titanic) dataset
""")

st.sidebar.write ("For more info, please contact:")

st.sidebar.write("<a href='https://www.linkedin.com/in/yong-poh-yu/'>Dr. Yong Poh Yu </a>", unsafe_allow_html=True)




test_data_ratio = st.sidebar.slider('Select testing size or ratio', 
                                    min_value= 0.10, 
                                    max_value = 0.50,
                                    value=0.2)
n_estimators = st.sidebar.slider('Choose number of trees', 1, 1000,value=100)
max_depth = st.sidebar.slider('Choose number of levels', 1, 30,value=10)


titanic_data = pd.read_csv('titanic.csv')
titanic_data = titanic_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
titanic_data = titanic_data.dropna()

from sklearn.preprocessing import LabelEncoder

labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()

titanic_data['Sex'] = labelencoder1.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = labelencoder2.fit_transform(titanic_data['Embarked'])


X = titanic_data.drop('Survived',axis=1)
y = titanic_data['Survived']


from sklearn.model_selection import train_test_split


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=1234,test_size=test_data_ratio)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

RandomForest = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
RandomForest.fit(Xtrain, ytrain)
ypred = RandomForest.predict(Xtest)

st.write("The classification report (for testing set) is:")

# st.write(pd.DataFrame(confusion_matrix(ytest, ypred)))

report = classification_report(ytest, ypred,output_dict=True)
df = pd.DataFrame(report).transpose()
st.write(df)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
