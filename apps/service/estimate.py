from io import BytesIO
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
import numpy as np
import pydash as py_
from apps.service.csvProcess import initCSV
from .utils import convert_data_redmine, getUrlRedmine
from .common.constant import CONST_LABEL_ASSIGNEE, CONST_LABEL_CODING, CONST_LABEL_ESTIMATED_TIME, CONST_LABEL_EXPECTED, CONST_LABEL_ISSUE_NUMBER, CONST_LABEL_MOD, CONST_LABEL_NEW, CONST_LABEL_TRANSLATION, FILE_NAME, CONST_LABEL_TRAIN, CONST_LABEL_TRACKER_FROM_INPUT, CONST_LABEL_NEW_MOD_FROM_INPUT, FILE_NAME_JOB_LIB, MIN_ESTIMATE_TIME
# from sklearn.preprocessing import StandardScaler , MinMaxScaler
import cufflinks as cf
import plotly.io as pio
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple
from openpyxl import load_workbook

labelTrain = CONST_LABEL_TRAIN


def estimate(data):
    transfer = transferDataFromRequest(data)
    return loadModel(
        transfer,
        transfer[CONST_LABEL_TRACKER_FROM_INPUT],
        transfer[CONST_LABEL_NEW_MOD_FROM_INPUT])


def estimateCSVProcess(file):
    df = initCSV(file, False)
    DF_CODING_NEW = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_CODING)
                       & (df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_NEW)]
    DF_CODING_MOD = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_CODING)
                       & (df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_MOD)]
    DF_TRANSLATION_NEW = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_TRANSLATION) & (
        df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_NEW)]
    DF_TRANSLATION_MOD = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_TRANSLATION) & (
        df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_MOD)]
    predictions = []

    # add one value '#' to LabelTrain
    labelTrain_csv = labelTrain.copy()
    labelTrain_csv.append(CONST_LABEL_ISSUE_NUMBER)
    # Check DF is empty or not
    if not DF_CODING_NEW.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_CODING,
                    CONST_LABEL_NEW)))
        for index, row in DF_CODING_NEW[labelTrain_csv].iterrows():
            obj = {
                CONST_LABEL_ISSUE_NUMBER: str(
                    row[CONST_LABEL_ISSUE_NUMBER]).split('.')[0],
                'prediction': ''}
            row.drop(CONST_LABEL_ISSUE_NUMBER, inplace=True)
            prediction = lm.predict(row.values.reshape(1, -1))
            obj['prediction'] = prediction[0][0] / \
                60 if prediction[0][0] / 60 > 0 else MIN_ESTIMATE_TIME
            predictions.append(obj)

    if not DF_CODING_MOD.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_CODING,
                    CONST_LABEL_MOD)))
        for index, row in DF_CODING_MOD[labelTrain_csv].iterrows():
            obj = {
                CONST_LABEL_ISSUE_NUMBER: str(
                    row[CONST_LABEL_ISSUE_NUMBER]).split('.')[0],
                'prediction': ''}
            row.drop(CONST_LABEL_ISSUE_NUMBER, inplace=True)
            prediction = lm.predict(row.values.reshape(1, -1))
            obj['prediction'] = prediction[0][0] / \
                60 if prediction[0][0] / 60 > 0 else MIN_ESTIMATE_TIME
            predictions.append(obj)

    if not DF_TRANSLATION_NEW.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_TRANSLATION,
                    CONST_LABEL_NEW)))
        for index, row in DF_TRANSLATION_NEW[labelTrain_csv].iterrows():
            obj = {
                CONST_LABEL_ISSUE_NUMBER: str(
                    row[CONST_LABEL_ISSUE_NUMBER]).split('.')[0],
                'prediction': ''}
            row.drop(CONST_LABEL_ISSUE_NUMBER, inplace=True)
            prediction = lm.predict(row.values.reshape(1, -1))
            obj['prediction'] = prediction[0][0] / \
                60 if prediction[0][0] / 60 > 0 else MIN_ESTIMATE_TIME
            predictions.append(obj)

    if not DF_TRANSLATION_MOD.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_TRANSLATION,
                    CONST_LABEL_MOD)))
        for index, row in DF_TRANSLATION_MOD[labelTrain_csv].iterrows():
            obj = {
                CONST_LABEL_ISSUE_NUMBER: str(
                    row[CONST_LABEL_ISSUE_NUMBER]).split('.')[0],
                'prediction': ''}
            row.drop(CONST_LABEL_ISSUE_NUMBER, inplace=True)
            prediction = lm.predict(row.values.reshape(1, -1))
            obj['prediction'] = prediction[0][0] / \
                60 if prediction[0][0] / 60 > 0 else MIN_ESTIMATE_TIME
            predictions.append(obj)

    return predictions


def loadModel(transfer, tracker, new_mod):
    Test = pd.DataFrame([transfer])

    lm = joblib.load(FILE_NAME_JOB_LIB(FILE_NAME(tracker, new_mod)))
    # Scaler
    # scaler = StandardScaler()
    # X_test_scaled = scaler.fit_transform(Test[labelTrain])
    predictions = lm.predict(Test[labelTrain])

    return predictions[0]


def trainModel():
    """
    Train model base on TRACKER and NEW_MOD
    Divide 4 files: coding_new_joblib, coding_mod_joblib, translation_new_joblib, translation_mod_joblib
    """
    ESTIMATE_UTC = pd.read_csv(os.getcwd() + '/resources/data.csv')
    ESTIMATE_UTC = ESTIMATE_UTC.drop_duplicates()
    result_effective = []
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

            rs1, rs2 = readCSVAndTrainModel(
                X, y, tracker, new_mod)
            result_effective.append(
                {
                    "coeff": rs1,
                    "json": rs2
                }
            )
    return result_effective


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
    result_effective = ''
    dict_result = {}
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
        sns.histplot(y_test - predictions)
        plt.scatter(y_test, predictions)

        effective = metrics.explained_variance_score(y_test, predictions)
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
            effective)
        result_effective = f"Effective_{tracker}_{new_mod} = " + str(effective)

        # fields effect values
        coeff_df = pd.DataFrame(
            lm.coef_.T,
            X.columns,
            columns=['Coefficient'])
        # print(f'COEFF_{tracker}_{new_mod}:', coeff_df)
        dict_result = coeff_df.to_dict(orient='index')
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
            os.getcwd(), 'apps', 'static', 'assets', 'images',
            f'chart_{tracker}_{new_mod}.png')
        pio.write_image(fig, directorySaveImg, format='png', engine='kaleido')
    except Exception as error:
        print(error)
    return result_effective, dict_result


def transferDataFromRequest(data):
    result = {}
    for key in data.keys():
        result[key] = data[key]
    return convert_data_redmine(result)


def drawBoxPlot():
    ESTIMATE_UTC = pd.read_csv(os.getcwd() + '/resources/data.csv')
    ESTIMATE_UTC = ESTIMATE_UTC.drop_duplicates()
    X = ESTIMATE_UTC[labelTrain]
    y = ESTIMATE_UTC[CONST_LABEL_EXPECTED]

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


def analyzeDataFromCSVWithSpentTime(file):
    df = initCSV(file, False)
    DF_CODING_NEW = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_CODING)
                       & (df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_NEW)]
    DF_CODING_MOD = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_CODING)
                       & (df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_MOD)]
    DF_TRANSLATION_NEW = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_TRANSLATION) & (
        df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_NEW)]
    DF_TRANSLATION_MOD = df[(df[CONST_LABEL_TRACKER_FROM_INPUT] == CONST_LABEL_TRANSLATION) & (
        df[CONST_LABEL_NEW_MOD_FROM_INPUT] == CONST_LABEL_MOD)]
    predictions = []

    # add one value '#' to LabelTrain
    labelTrain_csv = labelTrain.copy()
    labelTrain_csv.append(CONST_LABEL_ISSUE_NUMBER)
    labelTrain_csv.append(CONST_LABEL_ASSIGNEE)
    labelTrain_csv.append(CONST_LABEL_TRACKER_FROM_INPUT)
    labelTrain_csv.append(CONST_LABEL_EXPECTED)
    labelTrain_csv.append(py_.snake_case(CONST_LABEL_ESTIMATED_TIME))

    # Check DF is empty or not
    if not DF_CODING_NEW.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_CODING,
                    CONST_LABEL_NEW)))
        for index, row in DF_CODING_NEW[labelTrain_csv].iterrows():
            predictions.append(convertToDfAndPredict(row, lm))

    if not DF_CODING_MOD.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_CODING,
                    CONST_LABEL_MOD)))
        for index, row in DF_CODING_MOD[labelTrain_csv].iterrows():
            predictions.append(convertToDfAndPredict(row, lm))

    if not DF_TRANSLATION_NEW.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_TRANSLATION,
                    CONST_LABEL_NEW)))
        for index, row in DF_TRANSLATION_NEW[labelTrain_csv].iterrows():
            predictions.append(convertToDfAndPredict(row, lm))

    if not DF_TRANSLATION_MOD.empty:
        lm = joblib.load(
            FILE_NAME_JOB_LIB(
                FILE_NAME(
                    CONST_LABEL_TRANSLATION,
                    CONST_LABEL_MOD)))
        for index, row in DF_TRANSLATION_MOD[labelTrain_csv].iterrows():
            predictions.append(convertToDfAndPredict(row, lm))

    return predictions


def getObjFromRowData(row):
    return {
        CONST_LABEL_ISSUE_NUMBER: str(
            row[CONST_LABEL_ISSUE_NUMBER]).split('.')[0],
        'prediction': '',
        CONST_LABEL_ASSIGNEE: row[CONST_LABEL_ASSIGNEE],
        CONST_LABEL_TRACKER_FROM_INPUT: row[CONST_LABEL_TRACKER_FROM_INPUT],
        CONST_LABEL_EXPECTED: row[CONST_LABEL_EXPECTED] / 60,
        'user_estimate_time': row[py_.snake_case(CONST_LABEL_ESTIMATED_TIME)] 
    }


def convertToDfAndPredict(row: pd.Series, lm: LinearRegression):
    obj = getObjFromRowData(row)
    features = row[labelTrain]
    dataFramePredict = pd.DataFrame([features])
    dataFramePredict.columns = labelTrain
    prediction = lm.predict(dataFramePredict)
    obj['prediction'] = prediction[0][0] / \
        60 if prediction[0][0] / 60 > 0 else MIN_ESTIMATE_TIME
    obj['gap'] = round(
        (abs(
            obj[CONST_LABEL_EXPECTED] -
            obj['prediction']) /
            obj[CONST_LABEL_EXPECTED]),
        2) if obj[CONST_LABEL_EXPECTED] != 0 else 0
    obj[py_.snake_case(CONST_LABEL_ESTIMATED_TIME)] = obj['user_estimate_time']
    return obj


def exportExcelReportGap(data: list):
    wb = load_workbook(
        filename=os.path.join(
            os.getcwd(),
            'resources',
            'template-report-gap.xlsx'))
    ws = wb.active
    start_row = 2
    for row_data in data:
        cell1 = ws.cell(
            row=start_row,
            column=1,
            value=row_data[CONST_LABEL_ISSUE_NUMBER])
        cell1.hyperlink = getUrlRedmine(row_data[CONST_LABEL_ISSUE_NUMBER])
        cell1.style = 'Hyperlink'

        ws.cell(row=start_row, column=2, value=row_data['prediction'])
        ws.cell(row=start_row, column=3, value=row_data[py_.snake_case(CONST_LABEL_ESTIMATED_TIME)])
        ws.cell(row=start_row, column=4, value=row_data[CONST_LABEL_EXPECTED])
        ws.cell(row=start_row, column=5, value=row_data[CONST_LABEL_TRACKER_FROM_INPUT])
        ws.cell(row=start_row, column=6, value=f"=ABS(B{start_row}-C{start_row})/C{start_row}")
        ws.cell(row=start_row, column=7, value=row_data['gap'])
        ws.cell(row=start_row, column=8, value=f"=ABS(B{start_row} - D{start_row})")
        

        ws.cell(row=start_row, column=9, value=row_data[CONST_LABEL_ASSIGNEE])
        start_row += 1  # next row

    output = BytesIO()
    wb.save(output)
    # Return value IOByte
    return output
