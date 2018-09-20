import os, re
import pandas
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

data_file = 'Hitters_cleaned.csv'



def read_csv(file, y_column):
    df = pandas.read_csv(file)
    y = df[y_column].values
    y_label = y_column
    df = df.drop(columns=[y_column])
    x = df.values
    x_labels = df.columns
    return x, y, x_labels, y_label

def create_plot(CLF, X, Y, alphas, title, labels):
    # based on http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html

    coefs = []
    for a in alphas:
        clf = CLF(a)
        clf.fit(X, Y)
        coefs.append(clf.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(np.log(alphas), coefs)
    plt.figlegend(labels)
    plt.xlabel('log(alpha)')
    plt.ylabel('weights')
    plt.title(title)
    plt.axis('tight')
    plt.show()


X, Y, x_labels, y_label = read_csv(data_file, 'Salary')
# print(x_labels)
# print(X)
# print(y_label)
# print(Y)
test_alphas = [1,3,10,30,100,300,1000,3000,10000,30000,100000]

clf = linear_model.LassoCV()
clf.fit(X, Y)
print(x_labels)
print('Lasso',clf.coef_)
print('Score', clf.score(X, Y))
create_plot(linear_model.Lasso, X, Y, test_alphas, 'Lasso', x_labels)

plot = linear_model.lasso_path(X,Y)
# print(plot)

clf = linear_model.RidgeCV()
clf.fit(X, Y)
print('Ridge',clf.coef_)
print('Score', clf.score(X, Y))
create_plot(linear_model.Ridge, X, Y, test_alphas, 'Ridge', x_labels)




