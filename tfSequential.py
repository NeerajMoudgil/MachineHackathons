from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

training_data= pd.read_excel("Data_Train.xlsx")
test_data= pd.read_excel("Test_set.xlsx")

print(training_data.head())
print(test_data.head())

tf.compat.v1.enable_eager_execution()


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
    text = int(hrs)
    return text

def stops(x):
  if(x=='non-stop'):
    x=int(0)
  else:
    x.strip()
    stps=x.split(' ')[0]
    x=stps
  return int(x)
#drop NA
training_data = training_data.dropna()
test_data = test_data.dropna()


#Change source destination values Delhi -> New Delhi

training_data['Source'] = training_data['Source'].apply(changeDelhiToNewDelhi)
training_data['Destination'] = training_data['Destination'].apply(changeDelhiToNewDelhi)

test_data['Source'] = test_data['Source'].apply(changeDelhiToNewDelhi)
test_data['Destination'] = test_data['Destination'].apply(changeDelhiToNewDelhi)


#Split dates to different columns

training_data['Journey_Day'] = pd.to_datetime(training_data.Date_of_Journey, format='%d/%m/%Y').dt.day
training_data['Journey_Month'] = pd.to_datetime(training_data.Date_of_Journey, format='%d/%m/%Y').dt.month


test_data['Journey_Day'] = pd.to_datetime(test_data.Date_of_Journey, format='%d/%m/%Y').dt.day
test_data['Journey_Month'] = pd.to_datetime(test_data.Date_of_Journey, format='%d/%m/%Y').dt.month

training_data.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
test_data.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)


training_data['Depart_Time_Hour'] = pd.to_datetime(training_data.Dep_Time).dt.hour
training_data['Depart_Time_Minutes'] = pd.to_datetime(training_data.Dep_Time).dt.minute

training_data.drop(labels = 'Dep_Time', axis = 1, inplace = True)


training_data['Arr_Time_Hour'] = pd.to_datetime(training_data.Arrival_Time).dt.hour
training_data['Arr_Time_Minutes'] = pd.to_datetime(training_data.Arrival_Time).dt.minute

training_data.drop(labels = 'Arrival_Time', axis = 1, inplace = True)


test_data['Depart_Time_Hour'] = pd.to_datetime(test_data.Dep_Time).dt.hour
test_data['Depart_Time_Minutes'] = pd.to_datetime(test_data.Dep_Time).dt.minute


test_data.drop(labels = 'Dep_Time', axis = 1, inplace = True)

test_data['Arr_Time_Hour'] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data['Arr_Time_Minutes'] = pd.to_datetime(test_data.Arrival_Time).dt.minute

test_data.drop(labels = 'Arrival_Time', axis = 1, inplace = True)


#Change duration text hr+min to minutes
training_data['Duration'] = training_data['Duration'].apply(changeDurationToMinutes)
test_data['Duration'] = test_data['Duration'].apply(changeDurationToMinutes)

training_data['Total_Stops']=training_data['Total_Stops'].apply(stops)
test_data['Total_Stops']=test_data['Total_Stops'].apply(stops)


print(training_data.head(),training_data.shape)
print(test_data.head(),test_data.shape)



CATEGORICAL_COLUMNS=['Airline','Source','Destination','Route','Additional_Info']
NUMERIC_COLUMNS=['Journey_Day','Journey_Month','Depart_Time_Hour','Depart_Time_Minutes','Total_Stops','Arr_Time_Hour','Arr_Time_Minutes','Duration']

# training_data.drop(labels = 'Route', axis = 1, inplace = True)
training_data.drop(labels = 'Additional_Info', axis = 1, inplace = True)

# test_data.drop(labels = 'Route', axis = 1, inplace = True)
test_data.drop(labels = 'Additional_Info', axis = 1, inplace = True)


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

le1 = LabelEncoder()

training_data['Route']=le1.fit_transform(training_data['Route'])

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

le1 = LabelEncoder()

test_data['Route']=le1.fit_transform(test_data['Route'])

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

print(training_data.shape)

training_data.drop('Price',axis=1,inplace=True)

print(X_train.shape)

# https://www.tensorflow.org/tutorials/keras/basic_regression

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(training_data.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

history = model.fit(
  X_train, Y_train,
  epochs=1000, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

test_predictions = model.predict(X_test).flatten()

print(sc_X.inverse_transform(test_predictions))
