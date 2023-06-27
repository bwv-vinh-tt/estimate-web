import pydash as py_
import os

root_dir = os.getcwd()

# common function


def convert_to_snake_case(arr):
    new_arr = []

    for string in arr:
        new_string = py_.snake_case(string)
        new_arr.append(new_string)

    return new_arr

# Contant variable


CONST_LABEL_FROM_REDMINE = '#,Parent task,Tracker,Subject,Spent time,Total estimated time,Total spent time,Assignee,Create Screen/API/Batch Qty,New/Mod,Doc Mod Quantity,Authority Qty,Display/Output Items Qty,Validation Items Qty,Event row,Event Items Qty,Get API Qty,Create API Qty,Update API Qty,Delete API Qty,Get Table Qty,Create Table Qty,Update Table Qty,Delete Table Qty,Doc Layout,Doc QA Amount,Doc Understandable,Doc File Format,Business Logic Level,Coding Method Level,Validation Qty,Field DB Qty'
CONST_LABEL_ESTIMATION_FIELD = 'Create Screen/API/Batch Qty,New/Mod,Doc Mod Quantity,Authority Qty,Display/Output Items Qty,Validation Items Qty,Validation Qty,Event row,Event Items Qty,Get API Qty,Create API Qty,Update API Qty,Delete API Qty,Get Table Qty,Create Table Qty,Update Table Qty,Delete Table Qty,Field DB Qty,Doc Layout,Doc QA Amount,Doc Understandable,Doc File Format,Business Logic Level,Coding Method Level'

CONST_LABEL_FROM_REDMINE_ARR = CONST_LABEL_FROM_REDMINE.split(',')

CONST_LABEL_ESTIMATION_FIELD_ARR = CONST_LABEL_ESTIMATION_FIELD.split(',')

CONST_LABEL_EXPECTED = 'spent_time'

CONST_LABEL_TRACKER_FROM_INPUT = 'tracker'

CONST_LABEL_NEW_MOD_FROM_INPUT = 'new_mod'

CONST_LABEL_CODING = 'Coding'

CONST_LABEL_TRANSLATION = 'Translation'

CONST_LABEL_NEW = 'New'

CONST_LABEL_MOD = 'Mod'

CONST_LABEL = {
    'LABEL001': 'Target coding task',
    'LABEL002': 'Target translation task',
}

CONST_REDMINE_URL = 'https://redmine.bridevelopment.com/issues/'

CONST_IGNORE_FIELD = [
    'Business Logic Level',
    'Coding Method Level',
    'Doc Understandable']


def FILE_NAME(tracker, new_mod):
    return "{}_{}_model.joblib".format(tracker, new_mod)


def FILE_NAME_JOB_LIB(file_name):
    return os.path.join(root_dir, 'joblib', file_name)


CONST_LABEL_TRAIN = list(
    filter(
        lambda x: x not in [
            'new_mod',
            'doc_file_format'],
        convert_to_snake_case(CONST_LABEL_ESTIMATION_FIELD_ARR)))

CONST_CHART_IMAGE_NAME = [
    'chart_Coding_New.png',
    'chart_Coding_Mod.png',
    'chart_Translation_New.png',
    'chart_Translation_Mod.png',
]
