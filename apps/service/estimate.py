import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
import joblib
import numpy as np
from .utils import convert_data_redmine
from .common.constant import CONST_LABEL_CODING, CONST_LABEL_EXPECTED, CONST_LABEL_MOD, CONST_LABEL_NEW, CONST_LABEL_TRANSLATION, FILE_NAME, CONST_LABEL_TRAIN, CONST_LABEL_TRACKER_FROM_INPUT, CONST_LABEL_NEW_MOD_FROM_INPUT, FILE_NAME_JOB_LIB
# from sklearn.preprocessing import StandardScaler , MinMaxScaler
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly.io as pio
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple

labelTrain = CONST_LABEL_TRAIN
pio.kaleido.scope.mathjax = None

def estimate(data):
    transfer = transferDataFromRequest(data)
    print(transfer)
    return loadModel(
        transfer,
        transfer[CONST_LABEL_TRACKER_FROM_INPUT],
        transfer[CONST_LABEL_NEW_MOD_FROM_INPUT])


def loadModel(transfer, tracker, new_mod):
    Test = pd.DataFrame([transfer])

    lm = joblib.load(FILE_NAME_JOB_LIB(FILE_NAME(tracker, new_mod)))
    # Scaler
    # scaler = StandardScaler()
    # X_test_scaled = scaler.fit_transform(Test[labelTrain])
    predictions = lm.predict(Test[labelTrain])

    print("Est time: ", pd.DataFrame(predictions, columns=['ouput']))
    return predictions[0]


def trainModel():
    """
    Train model base on TRACKER and NEW_MOD
    Divide 4 files: coding_new_joblib, coding_mod_joblib, translation_new_joblib, translation_mod_joblib
    """
    ESTIMATE_UTC = pd.read_csv(os.getcwd() + '/resources/data.csv')

    # remove outliers
    # ESTIMATE_UTC["validation_items_qty"] = BoxplotOutlierClipper().fit_transform(
    # ESTIMATE_UTC["validation_items_qty"])

    # ESTIMATE_UTC["display_output_item_qty"] = BoxplotOutlierClipper().fit_transform(
    # ESTIMATE_UTC["validation_items_qty"])

    # ESTIMATE_UTC["business_logic_level"] = BoxplotOutlierClipper().fit_transform(
    # ESTIMATE_UTC["business_logic_level"])

    # ESTIMATE_UTC["coding_method_level"] = BoxplotOutlierClipper().fit_transform(
    # ESTIMATE_UTC["coding_method_level"])

    for tracker in [CONST_LABEL_CODING, CONST_LABEL_TRANSLATION]:
        for new_mod in [CONST_LABEL_NEW, CONST_LABEL_MOD]:
            df = ESTIMATE_UTC[(ESTIMATE_UTC[CONST_LABEL_TRACKER_FROM_INPUT] == tracker) & (
                ESTIMATE_UTC[CONST_LABEL_NEW_MOD_FROM_INPUT] == new_mod)]
            if (check_df_can_train(df) == False):
                continue
            X = df[labelTrain]
            y = df[[CONST_LABEL_EXPECTED]]
            # Scaler
            # scaler = StandardScaler()
            # X_scaled = scaler.fit_transform(X)
            readCSVAndTrainModel(X, y, tracker, new_mod)
    return True


def check_df_can_train(df: pd.DataFrame):
    """
        Check df can train or not
    """
    min_required = 3
    if df is None or df.empty:
        return False
    if len(df.count()[df.count() >= min_required].index) == len(df.columns):
        return True
    else:
        return False


def readCSVAndTrainModel(
        X: pd.DataFrame,
        y: pd.DataFrame,
        tracker: str,
        new_mod: str):
    """
        Train by lm or Lasso
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=10)

        lm = LinearRegression()
        lm.fit(X_train, y_train)

        # lasso_model = Lasso(alpha=0.1)
        # lasso_model.fit(X_train, y_train)

        joblib.dump(lm, FILE_NAME_JOB_LIB(FILE_NAME(tracker, new_mod)))
        # StandardScaler apply  X_test
        # X_test_scaled = scaler.transform(X_test)
        # predictions = lm.predict(X_test_scaled)
        predictions = lm.predict(X_test)
        print(f"predicted response:\n{predictions}")
        sns.histplot(y_test - predictions)
        plt.scatter(y_test, predictions)

        # print('MAE:', metrics.mean_absolute_error(y_test, predictions))
        # print('MSE:', metrics.mean_squared_error(y_test, predictions))
        print(
            f'RMSE_{tracker}_{new_mod}:',
            np.sqrt(
                metrics.mean_squared_error(
                    y_test,
                    predictions)))
        print(
            f"Effective_{tracker}_{new_mod}: ",
            metrics.explained_variance_score(
                y_test,
                predictions))

        # fields effect values
        coeff_df = pd.DataFrame(
            lm.coef_.T,
            X.columns,
            columns=['Coefficient'])
        print(f'COEFF_{tracker}_{new_mod}:', coeff_df)

        cf.go_offline()
        pio.renderers.default = "colab"
        df = pd.DataFrame(
            {"EST": np.array(predictions).flatten(), "Spent": np.array(y_test).flatten()})
        fig = df.iplot(
            kind='scatter',
            mode='markers',
            asFigure=True,
            color=[
                'green',
                'red'])
        # Save the chart as a PNG file
        directorySaveImg = os.path.join(
            os.getcwd(),
            'apps\\static\\assets\\images',
            f'chart_{tracker}_{new_mod}.png')
        pio.write_image(fig, directorySaveImg, format='png', engine='kaleido')
    except Exception as error:
        print(error)
    return True


def transferDataFromRequest(data):
    result = {}
    for key in data.keys():
        result[key] = data[key]
    return convert_data_redmine(result)


def drawBoxPlot():
    ESTIMATE_UTC = pd.read_csv(os.getcwd() + '/resources/data.csv')

    X = ESTIMATE_UTC[labelTrain]
    y = ESTIMATE_UTC['spent_time']

    ESTIMATE_UTC.boxplot(column=labelTrain)
    plt.show()


def find_boxplot_boundaries(
    col: pd.Series, whisker_coeff: float = 1.5
) -> Tuple[float, float]:
    """Findx minimum and maximum in boxplot.

    Args:
        col: a pandas serires of input.
        whisker_coeff: whisker coefficient in box plot
    """
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - whisker_coeff * IQR
    upper = Q3 + whisker_coeff * IQR
    return lower, upper


class BoxplotOutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, whisker_coeff: float = 1.5):
        self.whisker = whisker_coeff
        self.lower = None
        self.upper = None

    def fit(self, X: pd.Series):
        self.lower, self.upper = find_boxplot_boundaries(X, self.whisker)
        return self

    def transform(self, X):
        return X.clip(self.lower, self.upper)
