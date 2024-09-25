import os

# import cv2
# from pytesseract import pytesseract, Output
import numpy as np
import pandas as pd
import pydash as py_

from .common.constant import (
    CONST_CHART_IMAGE_NAME,
    CONST_LABEL_CHECKED_ESTIMATION_ITEMS_VALUE_NO,
    CONST_LABEL_CHECKED_ESTIMATION_ITEMS_VALUE_YES,
    CONST_LABEL_EXPECTED,
    CONST_REDMINE_URL,
    root_dir,
)


def convert_data_redmine(row):
    result = py_.clone_deep(row)
    # CONVERT DIFF by some rules such as Coding / Mod ...
    #     DIFFICULT_NUM = 3 if py_.lower_case(result['tracker']) == 'coding' else 1
    #     result['new_mod'] = 0.3 if result['new_mod'] == 'New' else 0.2
    #     DIFFICULT_NUM = DIFFICULT_NUM * (1 + result['new_mod'])
    #     result['new_mod'] = result['new_mod'] * 10 * DIFFICULT_NUM

    # convert minutes
    if hasattr(result, CONST_LABEL_EXPECTED):
        result[CONST_LABEL_EXPECTED] = (
            getValueZeroIfNan(result[CONST_LABEL_EXPECTED]) * 60
        )
    result["create_scr_api_b_r_qty"] = getValueZeroIfNan(
        result["create_scr_api_b_r_qty"]
    )
    result["doc_mod_quantity"] = getValueZeroIfNan(result["doc_mod_quantity"])
    result["authority_qty"] = getValueZeroIfNan(result["authority_qty"])
    result["display_output_items_qty"] = getValueZeroIfNan(
        result["display_output_items_qty"]
    )
    result["validation_items_qty"] = getValueZeroIfNan(result["validation_items_qty"])
    result["event_row"] = getValueZeroIfNan(result["event_row"])
    result["event_items_qty"] = getValueZeroIfNan(result["event_items_qty"])
    result["get_api_qty"] = getValueZeroIfNan(result["get_api_qty"])
    result["create_api_qty"] = getValueZeroIfNan(result["create_api_qty"])
    result["update_api_qty"] = getValueZeroIfNan(result["update_api_qty"])
    result["delete_api_qty"] = getValueZeroIfNan(result["delete_api_qty"])
    result["get_table_qty"] = getValueZeroIfNan(result["get_table_qty"])
    result["create_table_qty"] = getValueZeroIfNan(result["create_table_qty"])
    result["update_table_qty"] = getValueZeroIfNan(result["update_table_qty"])
    result["delete_table_qty"] = getValueZeroIfNan(result["delete_table_qty"])
    if result["doc_layout"] == "Non-BW(Readable)":
        result["doc_layout"] = 15
    elif result["doc_layout"] == "Non-BW(Not Readable)":
        result["doc_layout"] = 30
    else:
        result["doc_layout"] = 10

    result["doc_qa_amount"] = getValueZeroIfNan(result["doc_qa_amount"])
    result["doc_understandable"] = py_.replace(result["doc_understandable"], "%", "")
    result["doc_understandable"] = (
        100 if result["doc_understandable"] == "nan" else result["doc_understandable"]
    )
    result["doc_understandable"] = 101 - py_.parse_int(result["doc_understandable"], 10)

    # convert doc_file_format -> maybe remove later
    # if (result['doc_file_format'] == 'Google Sheet'):
    #     result['doc_file_format'] = 10
    # elif (result['doc_file_format'] == 'Excel'):
    #     result['doc_file_format'] = 15
    # elif (result['doc_file_format'] == 'Other'):
    #     result['doc_file_format'] = 20
    # else:
    #     result['doc_file_format'] = 5

    result["business_logic_level"] = py_.replace(
        result["business_logic_level"], "%", ""
    )
    result["coding_method_level"] = py_.replace(result["coding_method_level"], "%", "")
    result["business_logic_level"] = (
        100
        if result["business_logic_level"] == "nan"
        else result["business_logic_level"]
    )
    result["business_logic_level"] = 101 - py_.parse_int(
        result["business_logic_level"], 10
    )
    result["coding_method_level"] = (
        100 if result["coding_method_level"] == "nan" else result["coding_method_level"]
    )
    result["coding_method_level"] = 101 - py_.parse_int(
        result["coding_method_level"], 10
    )
    result["validation_qty"] = getValueZeroIfNan(result["validation_qty"])
    result["field_db_qty"] = getValueZeroIfNan(result["field_db_qty"])
    return result


def getValueZeroIfNan(v):
    value = float(v)
    return 0 if pd.isna(value) or value == 0 or value == "0" else value


# def processImg(image):
#      #converting image into gray scale image
#      image_data = image.read()       # read the binary image data
#      nparr = np.fromstring(image_data, np.uint8)  # convert the binary data to a NumPy array
#      image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#      threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#      custom_config = r'--oem 3 --psm 6'
#      pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#      details = pytesseract.image_to_data(threshold_img, output_type=Output.DICT, config=custom_config, lang='eng')
#      parse_text = []
#      word_list = []
#      last_word = ''

#      for t in details['text']:
#           print(t)


#      print(details['text'])


def find_dict(array, condition):
    for dictionary in array:
        if condition(dictionary):
            return dictionary
    return None


def getUrlRedmine(id):
    if not str(id).isdigit():
        return ""
    return CONST_REDMINE_URL + str(id) if id != "" else ""


def get_chart_or_remove(removeFlag: bool = False):
    result = []
    for name in CONST_CHART_IMAGE_NAME:
        PATH_FILE = os.path.join(root_dir, "apps/static/assets/images", name)
        if os.path.exists(PATH_FILE):
            if removeFlag:
                os.remove(PATH_FILE)
            result.append(name)
    return result


def getCheckedEstimationFlagValue(value):
    if value in CONST_LABEL_CHECKED_ESTIMATION_ITEMS_VALUE_YES:
        return "Yes"
    elif value in CONST_LABEL_CHECKED_ESTIMATION_ITEMS_VALUE_NO:
        return "No"
    else:
        return None
