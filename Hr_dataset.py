import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pandas.read_csv("HR_comma_sep.csv")
# print(data)

#step 1-> checking missing data for any row and column

# print(data.isnull().values.any())


#step 2: check datatype

# print(data.dtypes)

#step 3 -> check unique values
# print(data.salary.unique())
# print(data.Department.unique())


clean_up_values = {
    "salary":{
        'low':1,
        'medium':2,
        'high':3
    }
}

data.replace(clean_up_values, inplace=True)
# print(data)

#step 4-> get dummies for the department and marge them in real data

dummies_data = pandas.get_dummies(data,columns=['Department'])
# print(dummies_data.columns)

#step 5-> delete unnecessary column if needed

final_data = dummies_data.drop(['Department_hr'], axis = 'columns') # deleting Department_hr column from data set
print(final_data.columns)

#step 6 -> draw model
# plt.scatter(x=final_data.salary, y= final_data.left)
# plt.scatter(x=final_data.satisfaction_level, y= final_data.left)
# plt.scatter(x=final_data.time_spend_company, y= final_data.left)
# plt.show()



X = final_data.drop(['left'], axis='columns')
y=final_data.left

# print(X.columns)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
model = LogisticRegression()
model.fit(X_train,y_train)
accuracy = model.score(X_test,y_test)
print(accuracy)

# modelr = RandomForestClassifier()
# modelr.fit(X_train,y_train)
# accuracy2 = modelr.score(X_test,y_test)
# print(accuracy2)


result1 = model.predict([[0.92,0.51,2,160,3,1,1,1,0,0,1,1,0,0,1,1,1]])

if(result1[0]==0):
    print("Employee won't leave")
else:
    print("Employee will leave")

# result2 = modelr.predict([[0.92,0.51,2,160,3,1,1,1,0,0,1,1,0,0,1,1,1]])
# print(result2)
