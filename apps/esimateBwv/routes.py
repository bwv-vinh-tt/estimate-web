
import json
from apps.esimateBwv import blueprint
from flask import Response, abort, render_template, request
from flask_login import login_required
from apps.service.common.constant import CONST_LABEL_CHECKED_ESTIMATION_ITEMS, MIN_ESTIMATE_TIME
from apps.service.csvProcess import checkIfAnyFieldEmptyInDict, compareEstimateFieldFromCSV, initCSV

from apps.service.estimate import analyzeDataFromCSVWithSpentTime, drawBoxPlot, estimate, estimateCSVProcess, exportExcelReportGap, trainModel
from apps.service.utils import getUrlRedmine, get_chart_or_remove
import urllib.parse
import os

# Check if directory exists
directory1 = os.path.join(os.getcwd(), 'apps', 'static', 'assets', 'images')
directory2 = os.path.join(os.getcwd(), 'resources')
directory3 = os.path.join(os.getcwd(), 'joblib')
if not os.path.exists(directory1):
    # Create the directory
    os.makedirs(directory1)
if not os.path.exists(directory2):
    # Create the directory
    os.makedirs(directory2)
if not os.path.exists(directory3):
    # Create the directory
    os.makedirs(directory3)
# End


@blueprint.route('/estimate', methods=('GET', 'POST'))
@login_required
def renderEstimate():

    return render_template('estimatePage/estimate.html', segment='estimate')


@blueprint.route("/importCSV", methods=['GET'])
@login_required
def importCSV():
    return render_template(
        "estimatePage/importCSV.html",
        segment='importCSV',
    )


@blueprint.route("/exportReport", methods=['GET'])
def exportReport():
    return render_template(
        "estimatePage/importCSV.html",
        segment='exportReport',
    )


@blueprint.route("/calc", methods=('GET', 'POST'))
def calc():
    if request.method == 'GET':
        return render_template('estimatePage/estimate.html')
    else:
        try:
            result = estimate(request.form)
            result = result[0] / 60 if result[0] / \
                60 > 0 else MIN_ESTIMATE_TIME
            data = {'result': result}
            return data
        except Exception as e:
            return {'error': 'Model not found'}


@blueprint.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['file']
    if not file:
        return abort(500, 'No file uploaded')
    elif not file.filename.endswith('.csv'):
        return 'File is not a CSV file'
    else:
        initCSV(file)
        get_chart_or_remove(True)
        effective = trainModel()
        img_array = get_chart_or_remove()
    return render_template(
        "estimatePage/importCSV.html",
        img=True,
        img_array=img_array,
        urllib_parse=urllib.parse,
        effective=effective)


@blueprint.route('/checkBoxPlot', methods=['POST'])
@login_required
def checkBoxPlot():
    file = request.files['file']
    if not file:
        return abort(500, 'No file uploaded')
    elif not file.filename.endswith('.csv'):
        return abort(500, 'File is not a CSV file')
    else:
        initCSV(file)
        drawBoxPlot()
    return render_template("estimatePage/importCSV.html", img=True)


@blueprint.route('/checkEstimateFieldValue', methods=['POST', 'GET'])
def checkEstimateFieldValue():
    try:
        if request.method == 'POST':
            file = request.files['file']
            if not file:
                return abort(500, 'No file uploaded')
            elif not file.filename.endswith('.csv'):
                return abort(500, 'File is not a CSV file')
            else:
                result = compareEstimateFieldFromCSV(request.files['file'])
                # sort by class red-text
                resultSorted = sorted(
                    result,
                    key=lambda k: (
                        k['class'] == '',
                        k[CONST_LABEL_CHECKED_ESTIMATION_ITEMS] != 'No'),
                )
                return render_template(
                    "estimatePage/compareEstimateFieldValue.html",
                    result=resultSorted,
                    getUrlRedmine=getUrlRedmine,
                    segment='checkEstimateFieldValue',
                    checkIfAnyFieldEmptyInDict=checkIfAnyFieldEmptyInDict)
        else:
            return render_template(
                'estimatePage/importCSVCompareItem.html',
                segment='checkEstimateFieldValue')
    except Exception as e:
        abort(500, e)


@blueprint.route('/estimateCSV', methods=['POST'])
@login_required
def estimateCSV():
    file = request.files['file']
    if not file:
        return abort(500, 'No file uploaded')
    elif not file.filename.endswith('.csv'):
        return abort(500, 'File is not a CSV file')
    else:
        result = estimateCSVProcess(request.files['file'])
        # sort list result by props '#' ASC
        result = sorted(result, key=lambda k: k['#'])

    return render_template(
        "estimatePage/estimateCSV.html",
        result=result,
        getUrlRedmine=getUrlRedmine,
    )


@blueprint.route('/reportResultFromCSV', methods=['POST'])
def reportResultFromCSV():
    file = request.files['file']
    if not file:
        return abort(500, 'No file uploaded')
    elif not file.filename.endswith('.csv'):
        return abort(500, 'File is not a CSV file')
    else:
        result = analyzeDataFromCSVWithSpentTime(request.files['file'])
        # sort list result by props '#' ASC
        result = sorted(result, key=lambda k: k['#'])

    return render_template(
        "estimatePage/reportResultFromCSV.html",
        result=result,
        getUrlRedmine=getUrlRedmine,
        serialized_list=json.dumps(result),
        segment='exportReport',
    )


@blueprint.route('/exportExcelReportGap', methods=['POST'])
def exportExcel():
    data: list = json.loads(request.form['data'])
    if len(data) == 0:
        return abort(500, 'No data')
    else:
        result = exportExcelReportGap(data)
        return Response(
            result.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': 'attachment; filename=data.xlsx'})
