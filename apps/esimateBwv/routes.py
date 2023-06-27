
from apps.esimateBwv import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
from apps.service.csvProcess import checkIfAnyFieldEmptyInDict, compareEstimateFieldFromCSV, initCSV

from apps.service.estimate import drawBoxPlot, estimate, trainModel
from apps.service.utils import getUrlRedmine, get_chart_or_remove
import urllib.parse


@blueprint.route('/estimate', methods=('GET', 'POST'))
def renderEstimate():

    return render_template('estimatePage/estimate.html', segment='estimate')


@blueprint.route("/importCSV", methods=('GET', 'POST'))
@login_required
def importCSV():
    return render_template("estimatePage/importCSV.html", segment='importCSV')


@blueprint.route("/calc", methods=('GET', 'POST'))
def calc():
    if request.method == 'GET':
        return render_template('estimatePage/estimate.html')
    else:
        result = estimate(request.form)
        data = {'result': result[0] / 60}
        return data


@blueprint.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['file']
    if not file:
        return 'No file uploaded'
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
        return 'No file uploaded'
    elif not file.filename.endswith('.csv'):
        return 'File is not a CSV file'
    else:
        initCSV(file)
        drawBoxPlot()
    return render_template("estimatePage/importCSV.html", img=True)


@blueprint.route('/checkEstimateFieldValue', methods=['POST', 'GET'])
def checkEstimateFieldValue():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return 'No file uploaded'
        elif not file.filename.endswith('.csv'):
            return 'File is not a CSV file'
        else:
            result = compareEstimateFieldFromCSV(request.files['file'])
            # sort by class red-text
            resultSorted = sorted(
                result, key=lambda k: k['class'], reverse=True)
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
