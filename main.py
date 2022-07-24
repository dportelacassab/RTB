# Hello and Welcome to the MediaForce data science test
# Name of Candidate: Diego Portela-Cassab
#
# Date: 17/07/22
#
# Time (Home test) - completing the test should take approximately 3-4 hours. If you see that it takes longer than that,
# you can write down your research plan and design in pseudo code or free speech.
#
# General information:
#
# We will be using the following parameters in order to grade your test:
# A clear explanation of your process and your results
# Clean, reusable and "easy to read" code
# Level of data processing and machine learning methodology.
# Creativity and innovation.
# If you have any questions regarding the test, please contact us at asafb@mediaforce.com and sharft@mediaforce.com
# In this test you will be implementing a solution to a classification problem. We recommend that you read the full test before you begin.
#
# the csv file contains events from a real time bidding (RTB) system. the data is composed of categorical variables containing labels,
# and a binary target variable indicating if a click has occurred.
#
# We wish to build a decision engine the decides if an advertisement should be presented to a certain user. each presentation of
# an ad costs 1 dollar and if a click is generated, we receive a payment of 45 dollars. we would like to decide for each new request
# if an ad should be presented on that request.
#
# Before we begin, use the following cell to perform all your imports. You can use any package you choose to if it can be installed
# via pip install
# write all your imports here: ##################################################################################################

import pandas as pd;
import numpy as np;
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing_ import Preprocessing_
from visualization_ import Visualization_
import os

if __name__ == '__main__':

    #################################################################################################################################
    # Data exploration ##############################################################################################################
    # Get to know your data. Use the following cell to show us how you get acquainted with new data and start a new project. you can add a few
    # cells if needed. # please add a clear documentation to help understand your thought process.

    # First of all I start by reading the data into the environment.
    path = '/Applications/Documents - מסמכים/BGU/MediaForce/mediaforce_test_data/';os.chdir(path);os.getcwd()
    data = pd.read_csv('mediaforce_test_data.csv', header=0);
    # Let's see how it looks
    print(data.head())

    d_ = data.values;          # saves the data as

    vas = list(data.columns)      # column names of the dataset variables
    row_ = data.shape[0]          # of rows
    col_ = data.shape[1]          # of columns
    data["click"] = data['click'].fillna(0)  # see that we have nan that represent when there is no click so I replace them with 0

    print('\n -------------------------  Data Exploration --------------------------------- \n')

    # let's see at the category types we have on each of the variables and if we can understand what they mean.
    # Variable 1 (X1) - ssp_id : I found some information on - https://protocol.bidswitch.com/_downloads/bda2dba30d33b3afc487d0138b758a21/BidSwitch_specs_5.3.pdf
    # Sell Side Platforms (SSPs).

    print('\n -------------------------  Variable X1 - ssp_id --------------------------------- \n')
    tr = data["ssp_id"].astype("category")
    ca1 = tr.cat.categories
    ca1.shape
    print( data["ssp_id"].astype("category") )
    # I found 13 categories for X1. One basic question we get here is which category represents the data the most. Let's
    # see the frecuencies for the 13 of them.
    print( data["ssp_id"].astype("category").value_counts()/row_ )
    # notice that Google, Bidswitch, medianet and triplelift represent more than 80% of the data.

    data_crosstabX1 = pd.crosstab(data['ssp_id'], data['click'], margins=False);
    print(data_crosstabX1)

    # (data_crosstabX1.iloc[2,1] + data_crosstabX1.iloc[0,1] + data_crosstabX1.iloc[4,1] + data_crosstabX1.iloc[11,1])/sum(data_crosstabX1.iloc[:,1])
    # and from these 80% of data, 96% of clicks are represent, so grouping the rest of the categories as one category may help
    # the model to increase accuracy.

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
    print('\n -------------------------  Variable X2 - publisher_type --------------------------------- \n')
    # Variable 2 (X2) - publisher_type :
    print(data["publisher_type"].astype("category"))
    # we have 2 categories: Website and Application.
    print( data["publisher_type"].astype("category").value_counts()/row_ )
    # notice that 63% of the users were navigating on a website and 36% on a mobile app.

    #for publisher id and domain no frecuencies are gonna be obtained given that each id's are supposed to be unique.
    #and seem to mimic information obtained from X1.

    print('\n -------------------------  Variable X3 - geo_region --------------------------------- \n')
    # Variable 3 (X3) - geo_region:
    cat3 = data["geo_region"].astype("category").cat.categories
    print( cat3 )
    # we notice the 50 states of the United States and and extra one which would be the state of Israel where the company focus
    # its main attention. #check this.
    ge3 = data["geo_region"].astype("category").value_counts() / row_
    print( ge3 )
    ge3.index
    # California, Texas, Florida, NY, Ilinois,Ohio, Pasadena, Michigan, North Carolina, Georgia, Tennessee, Washington, Arizona, this
    # first 12 states represent a total of 62% of the data for variable X3. None of the states itself has more than 10% representation
    # of the data.

    print('\n -------------------------  Variable X4 - device_os --------------------------------- \n')
    # Variable 4 (X4) - device_os: It specifies the Operating system used by the user
    print(data["device_os"].astype("category"))
    # we found 8 categories.
    print(data["device_os"].astype("category").value_counts() / row_)
    # Android, iOs, Windows which represent more than 80% themselves and OSX, ChromeOS, Linux, Blackberry and NetCast to complete the whole list.

    print('\n -------------------------  Variable X5 - device_type --------------------------------- \n')
    # Variable 5 (X5) - device_type:
    print(data["device_type"].astype("category"))
    # we found 4 categories. Phone, Tablet, PC and MediaCenter.
    print(data["device_type"].astype("category").value_counts() / row_)
    # we can see that MediaCenter becomes a variable that rarely appear with almost 0% of frecuency, and Phone with more than 50%.

    # Variable - placement type: which seem to deal with the 13 categories found in X1, but with some of them with internal variations
    # I mean, bidswitch has nativo, medianet, smaato, etc... so this variable could be left aside for mimicking information already found in X1
    # Similarly for variable mf_user_id will be treated similarly as publisher_id where each of them are supposed to be unique and then

    print('\n -------------------------  Variable Y - Click --------------------------------- \n')
    #Variable Y - click: click has ocurred or not.
    print(data["click"].astype("category"))
    print(data["click"].astype("category").value_counts() / row_)

    # and right after it I start the preprocessing, so I can clean, fill or delete blank spaces and deal with the dataset throuroughly

    #################################################################################################################################
    # Preprocessing #################################################################################################################
    # Please write a class or script that performs all the preprocessing needed before you start creating your predictive model.Normally,
    # we would break this process into multiple classes / scripts, but as this is a limited task, you can write a single class / function
    # \that performs all the preprocessing.

    print( '\n -------------------------  Preprocessing --------------------------------- \n' )

    pro_ = Preprocessing_(data);pro_.process_();
    data_prep = pro_.d
    print(data_prep.head())
    data_prep['device_os'].unique()

    # Independence between some of the variables is also checked by using a Pearson's Chi-Squared test
    pro_.independ_(data_prep.iloc[:,0:4])

    # Independent (fail to reject H0)     -     ssp_id   &   publisher_type
    # Independent (fail to reject H0)     -     ssp_id   &   geo_region
    # Independent (fail to reject H0)     -     ssp_id   &   device_os
    # Independent (fail to reject H0)     -     publisher_type   &   geo_region
    # Independent (fail to reject H0)     -     publisher_type   &   device_os
    # Independent (fail to reject H0)     -     geo_region   &   device_os
    # All of these variables are independent so a full model(all of the explanatory variables will be used from the beginning).

    # A usefull analysis to see some of association is given by combined contingency tables between some of the variables
    # Contingency table between geo_region, ssp_id and click.
    Region_Sspid_Click = pd.crosstab(index=data_prep["geo_region"],
                                 columns=[data_prep["ssp_id"],
                                          data_prep["click"]],
                                 margins=True)  # Include row and column totals
    Region_Sspid_Click
    # from this table we can see that category:
    # Google     is stronger by #clicks(1) on the states of  California(149) and Ilinois(150)
    # Bidswithc  is stronger by #clicks(1) on the states of  California(28)  and Florida(25)
    # Medianet   is stronger by #clicks(1) on the states of  California(21)  and Texas(19)
    # Triplelift is stronger by #clicks(1) on the states of  California(16), Ilinois(17) and Texas(17).
    # other      is stronger by #clicks(1) on the states of  California(9), Ilinois(11) and Florida(13).
    # we expect to see this as well on the visualization some of this spread of the data on the US territory with where we can see
    # which states present the highest amount of clicks.

    data_crosstab = pd.crosstab(data_prep['geo_region'], data_prep['click'], margins=False);print(data_crosstab)

    # Contingency table between publisher_type, device_type and clicks
    Publ_Dev_type_Click = pd.crosstab(index=data_prep["device_type"],
                                     columns=[data_prep["publisher_type"],
                                              data_prep["click"]],
                                     margins=True)
    Publ_Dev_type_Click
    # we can notice that in general MediaCenter has almost no impact on the data.
    # PC    has most of its clicks on website.
    # Phone has most of its clicks on Application.
    # Table has most of its clicks on website, however quite balanced contry to the two previous categories.

    #another possible useful analysis would be looking for insight if linux users prefer certain searcher engines and
    #if this shows more number of clicks, so we see the cross table for variables ssp_id, device_os and click
    Sspid_deviceType_Click = pd.crosstab(index=data_prep["device_os"],
                                      columns=[data_prep["ssp_id"],
                                               data_prep["click"]],
                                      margins=True)
    Sspid_deviceType_Click
    # first of all we notice that we can recategorize the BlackBerry, Netcast, ChromeOS and Linux  into just one category in variable device_os
    # in order to have more compact and spreaded categories. From this new category called Linux&Others we see that the clicks are found
    # mostly on Google with just 10 clicks.
    # For Android users clicks are mostly on Google(1748) followed by medianet(144)
    # For OSX and iOS woulnd be that bad to group the together and get a new one called iOS_general, this category has most of its clicks
    # on Google(567) followed by Bandwistch(119).



    pro_.final_()
    data_final = pro_.d
    print( data_final.head() )

    print('\n ------------------------- Visualization  --------------------------------- \n')
    # Visualization
    # Now it is time to present your initial understanding of the data to the product team. Please write a class for data presentation.
    # Make sure nothing is hard coded and this is a generic class that will work with any data you feed into it. Write a separate method
    # for each visualization, provide at least 2 methods.

    vis = Visualization_(data_prep);
    vis.countPlot();
    vis.map();

    # PLease for the modeling section, I used main3.py where I worked on it as a very independent script














    ####################################################################################################################
    #          Multiple Correspondence Analysis            #############################################################
    #####################################################################################################################
    # import prince
    #
    # mca = prince.MCA(n_components=2, n_iter=3, copy=True, check_input=True, engine='auto', random_state=42)
    # mca = mca.fit(data_prep)
    # mca.explained_inertia_
    #
    # ax = mca.plot_coordinates(X = data_prep,
    #                           ax= None,figsize=(6,6),show_row_points=True,
    #                           row_points_size=10,show_column_points=True,
    #                           column_points_size=30,show_column_labels=True,legend_n_cols=1)
    # ax.get_figure().savefig('mca.pdf')
    # mca.eigenvalues_
    # mca.total_inertia_
    # mca.explained_inertia_










    # print('\n ------------------------- Modeling  --------------------------------- \n')
    # #Modeling
    # #Use the next cell to create your model.
    # #Present your results Use the next cell to perform your analysis.
    # # report the results and present them verbally and graphically. here you should describe:
    # # which performance metric you used and why?
    # # model validation score (how you know the model is good).
    # # prediction validation score (how you know the prediction is good).
    # #
    #
    # X = data_final.loc[:, data_final.columns != 'click']
    # y = data_final.loc[:, data_final.columns == 'click']
    #
    # # Over-sampling using SMOTE
    # from imblearn.over_sampling import SMOTE
    #
    # os_ = SMOTE(random_state=0)
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # columns = X_train.columns
    # os_data_X, os_data_y = os_.fit_resample(X_train, y_train)
    # os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    # os_data_y = pd.DataFrame(data=os_data_y, columns=['click'])
    # # we can Check the numbers of our data
    # print("length of oversampled data is ", len(os_data_X))
    # print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['click'] == 0]))
    # print("Number of subscription", len(os_data_y[os_data_y['click'] == 1]))
    # print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['click'] == 0]) / len(os_data_X))
    # print("Proportion of subscription data in oversampled data is ",   len(os_data_y[os_data_y['click'] == 1]) / len(os_data_X))
    #
    # # Recursive Feature Elimination
    # data_final_vars = data_final.columns.values.tolist()
    # y = ['click']
    # X = [i for i in data_final_vars if i not in y]
    # from sklearn.feature_selection import RFE
    # from sklearn.linear_model import LogisticRegression
    #
    # logreg = LogisticRegression()
    # rfe = RFE(logreg)
    # rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    # print(rfe.support_)
    # print(rfe.ranking_)
    #
    # cols = data_final.columns.values[np.where(rfe.support_ == False)]
    # cols = cols[0:len(cols)]
    # cols = ['ssp_id_bidswitch','ssp_id_google','ssp_id_medianet','ssp_id_other',
    #         'publisher_type_application','publisher_type_website',
    #         'device_os_Android','device_os_Linux&others','device_os_Windows','device_os_iOS_general',
    #         'device_type_MediaCenter','device_type_PC','device_type_Phone','device_type_Tablet']
    #
    # X = os_data_X[cols]
    # y = os_data_y['click']
    #
    # import statsmodels.api as sm
    #
    # logit_model = sm.Logit(y, X)
    # result = logit_model.fit()
    # # Singular Matrix, sooo I have to chose the variable to fit the model somehow...
    # # so then I get a result and continue with the process...
    # print(result.summary2())
    #
    #
    # from sklearn.linear_model import LogisticRegression
    # from sklearn import metrics
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # logreg = LogisticRegression()
    # logreg.fit(X_train, y_train)
    #
    # y_pred = logreg.predict(X_test)
    # print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
