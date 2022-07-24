# https://machinelearningmastery.com/imbalanced-classification-with-python-7-day-mini-course/

print('\n ------------------------- Modeling  --------------------------------- \n')
    #Modeling
    #Use the next cell to create your model.
    #Present your results Use the next cell to perform your analysis.
    # report the results and present them verbally and graphically. here you should describe:
    # which performance metric you used and why?
    # model validation score (how you know the model is good).
    # prediction validation score (how you know the prediction is good).


from keras.models import Sequential
from keras.layers import Dense
import os
import pandas as pd
import numpy as np
path = '/Applications/Documents - מסמכים/BGU/MediaForce/mediaforce_test_data/';os.chdir(path);os.getcwd()

data = pd.read_csv('mediaforce_test_data.csv', header=0);
data["click"] = data['click'].fillna(0)

data['ssp_id'].unique()
data['ssp_id'] = np.where(data['ssp_id'] == 'emx', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'sovrn', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'the33across', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'smaato', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'gothamads', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'vmx', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'nativo', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'outbrain', 'other', data['ssp_id'])
data['ssp_id'] = np.where(data['ssp_id'] == 'revcontent', 'other', data['ssp_id'])
data['ssp_id'].unique()

todelete = ["Unnamed: 0","publisher_id","domain","placement_type","mf_user_id"]
for val in todelete:
	print(val)
	if val in data.columns:
		data.pop(val)
data.dropna(inplace=True,axis=0)

data['device_os'].unique()
data['device_os'] = np.where(data['device_os'] == 'BlackBerry', 'Linux&others', data['device_os'])
data['device_os'] = np.where(data['device_os'] == 'ChromeOS', 'Linux&others', data['device_os'])
data['device_os'] = np.where(data['device_os'] == 'Linux', 'Linux&others', data['device_os'])
data['device_os'] = np.where(data['device_os'] == 'NetCast', 'Linux&others', data['device_os'])
data['device_os'] = np.where(data['device_os'] == 'OSX', 'iOS_general', data['device_os'])
data['device_os'] = np.where(data['device_os'] == 'iOS', 'iOS_general', data['device_os'])
data['device_os'].unique()

# data.describe()
vas = list(data.columns)
# for g in range(0, len(vas) - 1):  # So I don't categorize the Click variable
# 	data.iloc[g] = data.iloc[g].astype("category")

X = data.iloc[:,0:4]    ;  #X = pd.get_dummies(X, columns=vas)
X = X.astype("category")
X.dtypes
Xc = pd.get_dummies(X)
vasXc = list(Xc.columns)

y = data.iloc[:,5]
y = y.astype("category")
y.dtypes

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=0.3) # 70% training and 30% test






# https://www.analyticsvidhya.com/blog/2021/06/complete-guide-to-prevent-overfitting-in-neural-networks-part-1/
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
#       DATA AUGMENTATION      ########################################################################
#       Oversampling 		   ########################################################################

from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train[vasXc], y_train)
pd.Series(y_resampled).value_counts().plot(kind='bar', title='Class distribution after appying SMOTE', xlabel='click')

########################################################################################################################
#           Logistic Regression                 ########################################################################
########################################################################################################################

from sklearn.linear_model import LogisticRegression
log_reg_2 = LogisticRegression()
log_reg_3 = LogisticRegression(solver='liblinear', class_weight='balanced')

# Fit the model with the data that has been resampled with SMOTE
log_reg_2.fit(X_resampled, y_resampled)
# I get a convergenceWarning
# Predict on the test set (not resampled to obtain honest evaluation)

import statsmodels.api as sm

formula = 'click ~ '
log_reg_2_statsmodels = sm.Logit(y_resampled,X_resampled).fit()
print(log_reg_2_statsmodels.summary())


preds0 = log_reg_2.predict(X_test)

log_reg_2.predict([X_test.iloc[15,:]])
y_test.iloc[15]

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, preds0).ravel()
print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue positives: ', tp)

# There are four ways to check if the predictions are right or wrong:
# TN / True Negative: the case was negative and predicted negative
# TP / True Positive: the case was positive and predicted positive
# FN / False Negative: the case was positive but predicted negative
# FP / False Positive: the case was negative but predicted positive

confusion_matrix( y_test, preds0 )
from sklearn.metrics import classification_report
print(classification_report(y_test, preds0))


########################################################################################################################
# here you should describe:
########   which performance metric you used and why?   ################################################################

# ****
# Prediction accuracy is the most common metric for classification tasks, although it is inappropriate and potentially
# dangerously misleading when used on imbalanced classification tasks, which is our case, even though we try to balance it.
# Popular alternatives are the precision and recall scores that allow the performance of the model to be considered by
# focusing on the minority class, called the positive class.

# Precision is the ability of a classifier not to label an instance positive that is actually negative; Precision = TP/(TP + FP)
# Precision — What percent of your predictions were correct? # Precision = TruePositives / (TruePositives + FalsePositives)

#######      model validation score (how you know the model is good).   ################################################

# 4% of precision is just deplorable particularly for the category 1 which is the one that gives the returns to the enterprise.
# this model will make the company make wrong decision hence lose money.

# Recall predicts the ratio of the total number of correctly predicted positive examples divided by the total number of
# positive examples that could have been predicted. Maximizing recall will minimize false negatives.
# Recall = TruePositives / (TruePositives + FalseNegatives)

# i)  Recall of no click(0) went down from 0.99 to 0.68 meaning that there are more nonclicks that we did not succeed to find.
# ii) Recall of    click(1) went up   from 0.04 to 0.58 meaning that we succeeded to identify many more clicks.

############  prediction validation score (how you know the prediction is good).    ####################################
# The performance of a model can be summarized by a single score that averages both the precision and the recall, called
# the F-Measure. Maximizing the F-Measure will maximize both the precision and recall at the same time.
# F-measure = (2 * Precision * Recall) / (Precision + Recall)

# F1 Score : 0.08 is a very poor given that is far from the value 1, it tells us that the model does a poor job predicting
# whether or not there will will a click or not from the customer.


# Cross Validation in case it helps to see some variations of the scores we are getting.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
cv = KFold(n_splits=10, random_state=1, shuffle=True)
f1s = cross_val_score(log_reg_2, X_resampled, y_resampled, scoring='f1',cv=cv, n_jobs=8)   #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#here how to see f1 socres for both categories...
#see summary of the logistic model
np.mean(np.absolute(f1s))





# this regression is similar but given that the data was already balanced by SMOTE for all of the models
# it seems to not to have such of an impact
log_reg_3 = LogisticRegression(solver='liblinear', class_weight='balanced')
log_reg_3.fit(X_resampled, y_resampled)
preds3 = log_reg_3.predict(X_test)

log_reg_3.predict([X_test.iloc[26,:]])
y_test.iloc[26]


confusion_matrix( y_test, preds3 )
print(classification_report(y_test, preds3))
#                 precision    recall  f1-score   support
#          0.0       0.99      0.68      0.80     38214
#          1.0       0.04      0.59      0.08       886
#     accuracy                           0.68     39100
#    macro avg       0.51      0.64      0.44     39100
# weighted avg       0.96      0.68      0.79     39100



# I decided to add two extra models to see if I can improve the accuracy.

########################################################################################################################
#           Random Forest                       ########################################################################
########################################################################################################################

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)         #Create a Gaussian Classifier
# clf.fit(X_train,y_train)
clf.fit(X_resampled, y_resampled)

preds1=clf.predict(X_test)

from sklearn import metrics                          # Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, preds1))
clf.predict([X_test.iloc[26,:]])
y_test.iloc[26]

confusion_matrix( y_test, preds1 )
print(classification_report(y_test, preds1))

#                precision    recall  f1-score   support
#          0.0       0.99      0.62      0.76     38214
#          1.0       0.04      0.64      0.07       886
#     accuracy                           0.62     39100
#    macro avg       0.51      0.63      0.42     39100
# weighted avg       0.97      0.62      0.75     39100
# F1 score moved from 4% to 7%... this is still very poor


########################################################################################################################
#           Neural Networks                       ######################################################################
########################################################################################################################

model = Sequential()
model.add(Dense(10, input_dim=X_resampled.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])      # compile the keras model

model.fit(X_resampled, y_resampled, epochs=100, batch_size=16, verbose=2)              # fit the keras model on the dataset
# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

pred_=model.predict(X_test)

pred_[26]
y_test.iloc[26]

preds2 = (pred_ > 0.5).astype(int)
confusion_matrix(y_test, preds2)
print(classification_report(y_test, preds2))

#                 precision    recall  f1-score   support
#          0.0       0.99      0.62      0.76     38214
#          1.0       0.04      0.65      0.07       886
#     accuracy                           0.62     39100
#    macro avg       0.51      0.64      0.42     39100
# weighted avg       0.97      0.62      0.75     39100