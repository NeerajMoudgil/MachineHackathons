import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
arr1= np.array([1,2,3]).reshape(1,3).ravel()
arr2= np.array([3,4,5]).ravel()

print(arr1.shape)
print(arr2.shape)
print(np.dot(arr1,arr2))

training_data= pd.read_excel("Data_Train.xlsx")
test_data= pd.read_excel("Test_set.xlsx")

print(training_data.head())


def changeDelhiToNewDelhi(text):
    text = text.strip()
    if (text=="Delhi"):
        text= "New Delhi"
    text = str(text)
    return text

def changeDurationToMinutes(text):
    text = text.strip()
    total = text.split(' ')
    to = total[0]
    hrs = (int)(to[:-1]) * 60
    if ((len(total)) == 2):
        mint = (int)(total[1][:-1])
        hrs = hrs + mint
    text = str(hrs)
    return text


training_data = training_data.dropna()



training_data['Journey_Day'] = pd.to_datetime(training_data.Date_of_Journey, format='%d/%m/%Y').dt.day

training_data['Journey_Month'] = pd.to_datetime(training_data.Date_of_Journey, format='%d/%m/%Y').dt.month

training_data['Journey_Week_Day'] = pd.to_datetime(training_data.Date_of_Journey, format='%d/%m/%Y').dt.dayofweek


test_data['Journey_Day'] = pd.to_datetime(test_data.Date_of_Journey, format='%d/%m/%Y').dt.day

test_data['Journey_Month'] = pd.to_datetime(test_data.Date_of_Journey, format='%d/%m/%Y').dt.month

test_data['Journey_Week_Day'] = pd.to_datetime(test_data.Date_of_Journey, format='%d/%m/%Y').dt.dayofweek


training_data.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)

test_data.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)

training_data['Depart_Time_Hour'] = pd.to_datetime(training_data.Dep_Time).dt.hour
training_data['Depart_Time_Minutes'] = pd.to_datetime(training_data.Dep_Time).dt.minute

training_data.drop(labels = 'Dep_Time', axis = 1, inplace = True)


training_data['Arr_Time_Hour'] = pd.to_datetime(training_data.Arrival_Time).dt.hour
training_data['Arr_Time_Minutes'] = pd.to_datetime(training_data.Arrival_Time).dt.minute

training_data.drop(labels = 'Arrival_Time', axis = 1, inplace = True)
training_data.drop(labels = 'Additional_Info', axis = 1, inplace = True)



# Test Set


test_data['Depart_Time_Hour'] = pd.to_datetime(test_data.Dep_Time).dt.hour
test_data['Depart_Time_Minutes'] = pd.to_datetime(test_data.Dep_Time).dt.minute


test_data.drop(labels = 'Dep_Time', axis = 1, inplace = True)

test_data['Arr_Time_Hour'] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data['Arr_Time_Minutes'] = pd.to_datetime(test_data.Arrival_Time).dt.minute

test_data.drop(labels = 'Arrival_Time', axis = 1, inplace = True)
test_data.drop(labels = 'Additional_Info', axis = 1, inplace = True)

print(training_data.info())
print(test_data.info())

#training_data['Source'] = training_data['Source'].apply(changeDelhiToNewDelhi)
#training_data['Destination'] = training_data['Destination'].apply(changeDelhiToNewDelhi)

#test_data['Source'] = test_data['Source'].apply(changeDelhiToNewDelhi)
#test_data['Destination'] = test_data['Destination'].apply(changeDelhiToNewDelhi)

print(training_data.head())

print(training_data['Route'].values[0].strip().split('→'))

#vectorizer = HashingVectorizer(n_features=20)
vectorizer = CountVectorizer()

codes = []
identity= np.ones(20).reshape((20,1))
vectors = []


def changeRouteTextToVector(text):
    arr= text.split('→')
    for code in arr:
        code= code.strip()
        codes.append(code)
        code=[code]
        vectors.append(vectorizer.fit_transform(code).toarray().ravel())

    for vector in vectors:
        pass

    #np.linalg.multi_dot(vectors)
    return text

def stops(x):
  if(x=='non-stop'):
    x=str(0)
  else:
    x.strip()
    stps=x.split(' ')[0]
    x=stps
  return x
#training_data['Route'] = training_data['Route'].apply(changeRouteTextToVector)

# print(vectors[0])
# print(vectors[0].shape)
# print(vectors[1].shape)
# print(np.dot(vectors[0], vectors[1]))
# print(codes)

#print(identity.shape)

#categorical daTA
training_data_cat=training_data.select_dtypes(include=['object']).copy()



CATEGORICAL_COLUMNS=['Airline','Source','Destination','Route','Duration','Total_Stops','Additional_Info']
NUMERIC_COLUMNS=['Journey_Day','Journey_Month','Depart_Time_Hour','Depart_Time_Minutes','Arr_Time_Hour','Arr_Time_Minutes']
print(training_data_cat.head())
print(training_data_cat.isnull().sum())
print(training_data_cat['Airline'].value_counts())
print(training_data_cat['Source'].value_counts())
print(training_data_cat['Destination'].value_counts())
print(training_data_cat['Route'].value_counts())
print(training_data_cat['Duration'].value_counts())
print(training_data_cat['Total_Stops'].value_counts())

# found Trujet airline has only record so won't effect much

training_data=training_data[training_data.Airline != 'Trujet']

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

lb = LabelBinarizer()

#labelizing airlines
lb_results_airline = lb.fit_transform(training_data['Airline'])
lb_results_df_airline = pd.DataFrame(lb_results_airline, columns=lb.classes_)
training_data.drop(labels = 'Airline', axis = 1, inplace = True)
training_data = pd.concat([training_data.reset_index(drop=True), lb_results_df_airline.reset_index(drop=True)], axis=1)

#labelizing source
lb_results_Source = lb.fit_transform(training_data['Source'])
lb_results_df_Source = pd.DataFrame(lb_results_Source, columns=lb.classes_)
training_data.drop(labels = 'Source', axis = 1, inplace = True)
training_data = pd.concat([training_data.reset_index(drop=True), lb_results_df_Source.reset_index(drop=True)], axis=1)

#labelizing Destination
lb_results_Destination = lb.fit_transform(training_data['Destination'])
lb_results_df_Destination = pd.DataFrame(lb_results_Destination, columns=lb.classes_)
training_data.drop(labels = 'Destination', axis = 1, inplace = True)
training_data = pd.concat([training_data.reset_index(drop=True), lb_results_df_Destination.reset_index(drop=True)], axis=1)

#labelizing Total_Stops
# lb_results_Total_Stops = lb.fit_transform(training_data['Total_Stops'])
# lb_results_df_Total_Stops = pd.DataFrame(lb_results_Total_Stops, columns=lb.classes_)
# training_data.drop(labels = 'Total_Stops', axis = 1, inplace = True)
# training_data = pd.concat([training_data.reset_index(drop=True), lb_results_df_Total_Stops.reset_index(drop=True)], axis=1)

training_data['Total_Stops']=training_data['Total_Stops'].apply(stops)

le1 = LabelEncoder()

training_data['Route']=le1.fit_transform(training_data['Route'])
#training_data['Duration']=le1.fit_transform(training_data['Duration'])

training_data['Duration'] = training_data['Duration'].apply(changeDurationToMinutes)



print(training_data['Duration'])


print(training_data.head())
print(training_data.info())




print("\n Contains NaN/Empty cells : ", training_data[training_data.isnull().any(axis=1)])
print("\n Total empty cells by column :\n", training_data.isnull().sum())

#--- test data ---

#labelizing airlines
lb_results_airline = lb.fit_transform(test_data['Airline'])
lb_results_df_airline = pd.DataFrame(lb_results_airline, columns=lb.classes_)
test_data.drop(labels = 'Airline', axis = 1, inplace = True)
test_data = pd.concat([test_data.reset_index(drop=True), lb_results_df_airline.reset_index(drop=True)], axis=1)

#labelizing source
lb_results_Source = lb.fit_transform(test_data['Source'])
lb_results_df_Source = pd.DataFrame(lb_results_Source, columns=lb.classes_)
test_data.drop(labels = 'Source', axis = 1, inplace = True)
test_data = pd.concat([test_data.reset_index(drop=True), lb_results_df_Source.reset_index(drop=True)], axis=1)

#labelizing Destination
lb_results_Destination = lb.fit_transform(test_data['Destination'])
lb_results_df_Destination = pd.DataFrame(lb_results_Destination, columns=lb.classes_)
test_data.drop(labels = 'Destination', axis = 1, inplace = True)
test_data = pd.concat([test_data.reset_index(drop=True), lb_results_df_Destination.reset_index(drop=True)], axis=1)

#labelizing Total_Stops


# lb_results_Total_Stops = lb.fit_transform(test_data['Total_Stops'])
# lb_results_df_Total_Stops = pd.DataFrame(lb_results_Total_Stops, columns=lb.classes_)
# test_data.drop(labels = 'Total_Stops', axis = 1, inplace = True)
# test_data = pd.concat([test_data.reset_index(drop=True), lb_results_df_Total_Stops.reset_index(drop=True)], axis=1)

test_data['Total_Stops']=test_data['Total_Stops'].apply(stops)

le1 = LabelEncoder()

test_data['Route']=le1.fit_transform(test_data['Route'])
#test_data['Duration']=le1.fit_transform(test_data['Duration'])

test_data['Duration'] = test_data['Duration'].apply(changeDurationToMinutes)



print(test_data.tail())
print(test_data.info())

print("\n Contains NaN/Empty cells test_data : ", test_data[test_data.isnull().any(axis=1)])
print("\n Total empty cells by column test_data :\n", test_data.isnull().sum())


print("training data -----:;:::::",training_data.columns)
print("test_data data -----:;:::::",test_data.columns)


training_data.drop(labels = 'Route', axis = 1, inplace = True)
test_data.drop(labels = 'Route', axis = 1, inplace = True)

# Dependent Variable
Y_train = training_data.iloc[:,training_data.columns=='Price'].values  # 6 is the index of "Price" in the Training Set

# Independent Variables
X_train = training_data.iloc[:,training_data.columns != 'Price'].values # selects all columns except "Price"

# Independent Variables for Test Set
X_test = test_data.iloc[:,:].values

print("X trtain shape :::",X_train.shape)
print("X_test shape :::",X_test.shape)

print("xTrain dataframe",pd.DataFrame(X_train).describe())
print("training_data dataframe",training_data.describe())
print("xTrain columns",pd.DataFrame(X_train).columns)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
#Y_train = sc_y.fit_transform(Y_train)
Y_train = Y_train.reshape((len(Y_train), 1))

Y_train = sc_X.fit_transform(Y_train)

Y_train = Y_train.ravel()

print("xTrain",pd.DataFrame(X_train).describe())

print("yTrain",pd.DataFrame(Y_train).describe())
print("X_test",pd.DataFrame(X_test).describe())


print("training data -----:;:::::",training_data.columns)
print("test_data data -----:;:::::",test_data.columns)

# from sklearn.svm import SVR
#
# svr = SVR(kernel = "rbf")
#
# svr.fit(X_train,Y_train)
#
# Y_pred_svr = sc_X.inverse_transform(svr.predict(X_test))
#
#
# pd.DataFrame(Y_pred_svr, columns = ['Price']).to_excel("predictions_svr05.xlsx", index = False)

#svr07 with the best score with Random regressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1500, random_state = 42,oob_score=True,n_jobs=-1)
rf.fit(X_train,Y_train)
Y_pred = sc_X.inverse_transform(rf.predict(X_test))
pd.DataFrame(Y_pred, columns = ['Price']).to_excel("predictions_svr07.xlsx", index = False)

# feature_list = list(training_data.columns)
#
# # Get numerical feature importances
# importances = list(rf.feature_importances_)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # Print out the feature and importances
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]