# linear regression, logistic regression, decision tree, random forest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm
import numpy as np
import pickle


class Model:
    def __init__(self, train_X, train_Y):
        self.X = train_X
        self.Y = train_Y

    def lr_model_selection(self, solver, penalty, c_value, cross_val):
        model = LogisticRegression()
        grid = dict(solver=solver, penalty=penalty, C=c_value)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, cv=cross_val, scoring='accuracy')
        # return grid_search.fit(self.X, self.Y)
        grid_search_results = grid_search.fit(self.X, self.Y)
        print("Best: %f using %s" % (grid_search_results.best_score_, grid_search_results.best_params_))
        means = grid_search_results.cv_results_['mean_test_score']
        std = grid_search_results.cv_results_['std_test_score']
        params = grid_search_results.cv_results_['params']
        for mean, std, param in zip(means, std, params):
            print("%f (%f) with: %r" % (mean, std, param))

    def normalise(self, scale):
        self.normalised_Y = self.Y.apply(lambda x: x / scale)

    def lr_train(self, penalty='l2', C=1.0, solver='lbfgs', ):
        LR = LogisticRegression(penalty=penalty, C=C, solver=solver)
        LR.fit(self.X, self.Y)
        print("Training score:", LR.score(self.X, self.Y))
        self.LogsitcRegression = LR
        try:
            numpy_Y = np.array(self.normalised_Y, dtype=float)
            numpy_X = np.array(self.X, dtype=float)
            # logistic regression in statsmodels need labels range in [0,1]
            logit = sm.Logit(numpy_Y, numpy_X)
            analysis = logit.fit()
            print(analysis.summary2())
        except ValueError:
            print("Please normalise labels first.")

    def lr_model_prepare(self, penalty, C, solver, X, Y, filename):
        LR = LogisticRegression(penalty=penalty, C=C, solver=solver)
        LR.fit(X, Y)
        pickle.dump(LR, open(filename, 'wb'))
        print("The model trained is saved as '%s'." % filename)

    def lr_test(self, test_X, test_Y):
        LR = self.LogsitcRegression
        # y_pred = LR.predict(test_X)
        # print("check real Y and prediction:", test_Y,y_pred)
        print("Test Score:",LR.score(test_X, test_Y))