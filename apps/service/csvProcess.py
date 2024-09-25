import copy
import os

import numpy as np
import pandas as pd
import pydash as py_

from .common.constant import (
    CONST_IGNORE_FIELD,
    CONST_LABEL,
    CONST_LABEL_ASSIGNEE,
    CONST_LABEL_CHECKED_ESTIMATION_ITEMS,
    CONST_LABEL_CHECKED_ESTIMATION_ITEMS_VALUE_YES,
    CONST_LABEL_ESTIMATED_TIME,
    CONST_LABEL_ESTIMATION_FIELD,
    CONST_LABEL_EXPECTED,
    CONST_LABEL_FROM_REDMINE,
    CONST_LABEL_FROM_REDMINE_ARR,
    CONST_LABEL_ISSUE_NUMBER,
    CONST_LABEL_NEW_MOD_FROM_INPUT,
    CONST_LABEL_TRACKER_FROM_INPUT,
    CONST_REDMINE_URL,
)
from .utils import convert_data_redmine, find_dict, getCheckedEstimationFlagValue

# Get the directory path of the current script file
root_dir = os.getcwd()

label = [
    "#",
    "Tracker",
    "Spent time",
    "Create Scr/API/B/R Qty",
    "New/Mod",
    "Doc Mod Quantity",
    "Authority Qty",
    "Display/Output Items Qty",
    "Validation Items Qty",
    "Event row",
    "Event Items Qty",
    "Get API Qty",
    "Create API Qty",
    "Update API Qty",
    "Delete API Qty",
    "Get Table Qty",
    "Create Table Qty",
    "Update Table Qty",
    "Delete Table Qty",
    "Doc Layout",
    "Doc QA Amount",
    "Doc Understandable",
    "Doc File Format",
    "Business Logic Level",
    "Coding Method Level",
    "Checked Estimation Items",
]


class RedmineTask:
    def __init__(
        self,
        tracker="",
        spent_time=0,
        create_scr_api_b_r_qty=0,
        new_mod="",
        doc_mod_quantity=0,
        authority_qty=0,
        display_output_items_qty=0,
        validation_items_qty=0,
        validation_qty=0,
        event_row=0,
        event_items_qty=0,
        get_api_qty=0,
        create_api_qty=0,
        update_api_qty=0,
        delete_api_qty=0,
        get_table_qty=0,
        create_table_qty=0,
        update_table_qty=0,
        delete_table_qty=0,
        field_db_qty=0,
        doc_layout=0,
        doc_qa_amount=0,
        doc_understandable=0,
        #  doc_file_format=0,
        business_logic_level=0,
        coding_method_level=0,
    ):
        self.tracker = tracker
        self.spent_time = spent_time
        self.create_scr_api_b_r_qty = create_scr_api_b_r_qty
        self.new_mod = new_mod
        self.doc_mod_quantity = doc_mod_quantity
        self.authority_qty = authority_qty
        self.display_output_items_qty = display_output_items_qty
        self.validation_items_qty = validation_items_qty
        self.validation_qty = validation_qty
        self.event_row = event_row
        self.event_items_qty = event_items_qty
        self.get_api_qty = get_api_qty
        self.create_api_qty = create_api_qty
        self.update_api_qty = update_api_qty
        self.delete_api_qty = delete_api_qty
        self.get_table_qty = get_table_qty
        self.create_table_qty = create_table_qty
        self.update_table_qty = update_table_qty
        self.delete_table_qty = delete_table_qty
        self.field_db_qty = field_db_qty
        self.doc_layout = doc_layout
        self.doc_qa_amount = doc_qa_amount
        self.doc_understandable = doc_understandable
        # self.doc_file_format = doc_file_format
        self.business_logic_level = business_logic_level
        self.coding_method_level = coding_method_level

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"{self.spent_time}"


def initCSV(uploaded_file, isTrainingFile=True):
    CSV_IMPORT = pd.read_csv(uploaded_file)
    CSV_IMPORT = CSV_IMPORT.drop_duplicates()
    # remove record has CONST_LABEL_CHECKED_ESTIMATION_ITEMS != Yes
    if isTrainingFile:
        CSV_IMPORT = CSV_IMPORT[
            CSV_IMPORT[CONST_LABEL_CHECKED_ESTIMATION_ITEMS].isin(
                CONST_LABEL_CHECKED_ESTIMATION_ITEMS_VALUE_YES
            )
        ]

    CSV_IMPORT = CSV_IMPORT[CONST_LABEL_FROM_REDMINE_ARR]
    CSV_IMPORT.columns = py_.map_(
        CONST_LABEL_FROM_REDMINE_ARR, lambda v: py_.snake_case(v)
    )
    convert_data = []
    for index, row in CSV_IMPORT.iterrows():
        # remove bad data
        if (row[CONST_LABEL_EXPECTED] == 0 and isTrainingFile) or pd.isna(
            row[CONST_LABEL_NEW_MOD_FROM_INPUT]
        ):
            continue
        row = convert_data_redmine(row)
        new_record = RedmineTask()
        for col in label:
            col = "#" if py_.snake_case(col) == "" else py_.snake_case(col)
            if hasattr(new_record, col):
                new_record.__setitem__(col, row[col])
            if not isTrainingFile:
                new_record[CONST_LABEL_ISSUE_NUMBER] = row[""]
                new_record[CONST_LABEL_ASSIGNEE] = row[CONST_LABEL_ASSIGNEE]
                new_record[CONST_LABEL_TRACKER_FROM_INPUT] = row[
                    CONST_LABEL_TRACKER_FROM_INPUT
                ]
                new_record[py_.snake_case(CONST_LABEL_ESTIMATED_TIME)] = (
                    row[py_.snake_case(CONST_LABEL_ESTIMATED_TIME)]
                    if np.isnan(row[py_.snake_case(CONST_LABEL_ESTIMATED_TIME)])
                    == False
                    else 0
                )
        convert_data.append(new_record.__dict__)

    if isTrainingFile:
        # write csv
        df = pd.DataFrame(convert_data)
        file_path = os.path.join(root_dir, "resources", "data.csv")

        df.to_csv(file_path, index=False)
        return True
    else:
        return pd.DataFrame(convert_data)


def compareEstimateFieldFromCSV(uploaded_file):
    CSV_IMPORT = pd.read_csv(uploaded_file)
    CSV_IMPORT = CSV_IMPORT.drop_duplicates()

    # filter None data -> Empty string
    CSV_IMPORT = CSV_IMPORT.fillna("")
    # remove record has CONST_LABEL_CHECKED_ESTIMATION_ITEMS = '' from
    # CSV_IMPORT
    CSV_IMPORT = CSV_IMPORT[CSV_IMPORT[CONST_LABEL_CHECKED_ESTIMATION_ITEMS] != ""]
    CSV_IMPORT_GROUP_BY_PARENT_TASK = CSV_IMPORT.groupby(["Subject"])
    RESULT_RETURN = []
    result = {}
    # Parent task, target coding task, target coding task, difference item

    for label, group in CSV_IMPORT_GROUP_BY_PARENT_TASK:
        result[label] = group.to_dict(orient="records")

    initList = result.items()
    for key, value in initList:
        classRedText = ""
        keys_diff = None
        check = {}
        # test
        if value[0]["Parent task"] == 111072:
            print(value)
        # only compare if have 2 item in Parent task
        if len(value) == 2:
            for item in value:
                check = checkIfAnyFieldEmptyInDict(item)
                if check is not None:
                    classRedText = "red-text"

            keys_diff = compareCodingAndTranslateInOneGroupParentTask(
                value[0], value[1]
            )

            # if checkIfEstimationCheckIsNone(
            #         value[0]) is None and checkIfEstimationCheckIsNone(
            #         value[1]) is None:
            #     continue

            obj = {
                "Parent task": str(value[0]["Parent task"]).split(".")[0],
                CONST_LABEL["TARGET_CODING_TASK"]: getTrackerFromDict(
                    find_dict(value, lambda d: d["Tracker"] == "Coding")
                )[CONST_LABEL["TARGET_CODING_TASK"]],
                CONST_LABEL["TARGET_TRANSLATION_TASK"]: getTrackerFromDict(
                    find_dict(value, lambda d: d["Tracker"] == "Translation")
                )[CONST_LABEL["TARGET_TRANSLATION_TASK"]],
                "Difference item": (
                    ",".join([str(key) for key in keys_diff])
                    if keys_diff is not None
                    else ""
                ),
                "class": classRedText,
            }
            obj[CONST_LABEL_CHECKED_ESTIMATION_ITEMS] = (
                "No"
                if py_.get(obj, "Target coding task.Checked Estimation Items") == "No"
                or py_.get(obj, "Target translation task.Checked Estimation Items")
                == "No"
                else "Yes"
            )
            RESULT_RETURN.append(obj)
        else:
            for item in value:
                # if checkIfEstimationCheckIsNone(item) is not None:
                obj = {
                    "Parent task": str(value[0]["Parent task"]).split(".")[0],
                    CONST_LABEL["TARGET_CODING_TASK"]: getTrackerFromDict(item)[
                        CONST_LABEL["TARGET_CODING_TASK"]
                    ],
                    CONST_LABEL["TARGET_TRANSLATION_TASK"]: getTrackerFromDict(item)[
                        CONST_LABEL["TARGET_TRANSLATION_TASK"]
                    ],
                    "Difference item": (
                        ",".join([str(key) for key in keys_diff])
                        if keys_diff is not None
                        else ""
                    ),
                    "class": (
                        "red-text"
                        if checkIfAnyFieldEmptyInDict(item) is not None
                        else ""
                    ),
                }
                obj[CONST_LABEL_CHECKED_ESTIMATION_ITEMS] = (
                    "No"
                    if py_.get(obj, "Target coding task.Checked Estimation Items")
                    == "No"
                    or py_.get(obj, "Target translation task.Checked Estimation Items")
                    == "No"
                    else "Yes"
                )

                RESULT_RETURN.append(obj)
    return RESULT_RETURN


def compareCodingAndTranslateInOneGroupParentTask(dict1, dict2):
    label = copy.deepcopy(CONST_LABEL_ESTIMATION_FIELD.split(","))
    # compare 2 dict
    label = [item for item in label if item not in CONST_IGNORE_FIELD]
    values_diff = {
        k: dict1[k] for k in dict1 if k in dict2 and dict1[k] != dict2[k] and k in label
    }
    values_diff.update(
        {
            k: dict2[k]
            for k in dict2
            if k in dict1 and dict2[k] != dict1[k] and k in label
        }
    )

    return values_diff.keys() if len(values_diff) > 0 else None


def checkIfAnyFieldEmptyInDict(dict):
    label = copy.deepcopy(CONST_LABEL_ESTIMATION_FIELD.split(","))
    """Check if any field empty in dict OR conflict between New_Mod and Doc_mod_quantity"""
    if len(dict) == 0:
        return None

    tracker = dict["Tracker"]
    new_mod = dict["New/Mod"]
    check_conflict = (
        True
        if (new_mod == "New" and dict["Doc Mod Quantity"] != 0)
        or (new_mod == "Mod" and dict["Doc Mod Quantity"] == 0)
        else False
    )
    removeLabel = (
        "Coding Method Level" if tracker == "Translation" else "Business Logic Level"
    )
    if removeLabel in label:
        label.remove(removeLabel)
    for key, value in dict.items():
        if (value == "" and key in label) or check_conflict:
            return {
                "Key": key,
                CONST_LABEL["TARGET_CODING_TASK"]: value if tracker == "Coding" else "",
                CONST_LABEL["TARGET_TRANSLATION_TASK"]: (
                    value if tracker == "Translation" else ""
                ),
            }
    return None


def checkIfEstimationCheckIsNone(dict):
    if len(dict) == 0:
        return None

    flag = dict[CONST_LABEL_CHECKED_ESTIMATION_ITEMS]
    return None if flag == "" else flag


def getTrackerFromDict(dict):
    result = {
        CONST_LABEL["TARGET_TRANSLATION_TASK"]: {
            "Issue": "",
            "Data": {},
            "Assignee": "",
            CONST_LABEL_CHECKED_ESTIMATION_ITEMS: "",
        },
        CONST_LABEL["TARGET_CODING_TASK"]: {
            "Issue": "",
            "Data": {},
            "Assignee": "",
            CONST_LABEL_CHECKED_ESTIMATION_ITEMS: "",
        },
    }
    try:
        obj = {
            "Issue": dict["#"],
            "Data": dict,
            "Assignee": dict["Assignee"],
            CONST_LABEL_CHECKED_ESTIMATION_ITEMS: getCheckedEstimationFlagValue(
                dict[CONST_LABEL_CHECKED_ESTIMATION_ITEMS]
            ),
        }

        if dict["Tracker"] == "Translation":
            result[CONST_LABEL["TARGET_TRANSLATION_TASK"]] = obj
        else:
            result[CONST_LABEL["TARGET_CODING_TASK"]] = obj
    except BaseException:
        pass
    return result
