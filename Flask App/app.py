import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sklearn.externals import joblib
import traceback
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings("ignore")
from plotly import figure_factory as ff
from sklearn.metrics import roc_auc_score
from scipy import stats
import flask
import pickle
app = Flask(__name__)


def load_data():
	data=pd.read_csv(r"adult3.csv")
	return data


def create_histplot1(data):
	data = [go.Histogram(x=data["age"],marker=dict(color='rgb(255, 102, 102)'),opacity=0.75)]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Age',xref='paper',x=0),
	xaxis=dict(title="Age",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'),
    yaxis=dict(title='Count',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_histplot2(data):
	data = [go.Histogram(x=data["hours-per-week"],nbinsx=10,marker=dict(color='rgb(255, 102, 102)'),opacity=0.75 )]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Hours-per-week',xref='paper',x=0),
	xaxis=dict(title="Hours-per-week",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Count',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_barplot1(data):
	data = [go.Bar(x=['Married-civ-spouse' ,'Never-married' ,'Divorced', 'Separated','Widowed','Married-spouse-absent', 'Married-AF-spouse'],y=data["marital-status"].value_counts() ,marker=dict(color=['rgb(51, 153, 255)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)'],opacity=0.75))]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Marital-status',xref='paper',x=0),
	xaxis=dict(title="Marital-status",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_barplot2(data):
	data = [go.Bar(x=['Husband','Not-in-family', 'Own-child' ,'Unmarried', 'Wife', 'Other-relative'],y=data["relationship"].value_counts(),
		marker=dict(color=['rgb(51, 153, 255)','rgb(0, 70, 150)','rgb(0,110,230)','rgb(255, 102, 102)','rgb(0, 40, 100)','rgb(255, 102, 102)'],opacity=0.75))]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Relationship',xref='paper',x=0),
	xaxis=dict(title="Relationship",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_histplot3(data):
	data = [go.Histogram(x=data["capital-gain"],nbinsx=10,marker=dict(color='rgb(0, 120, 0)'),opacity=0.75 )]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Capital-Gain',xref='paper',x=0),
	xaxis=dict(title="Capital-Gain",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Count',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_barplot3(data):
	data = [go.Bar(x=['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm','10th' ,'7th-8th','Prof-school' ,'9th'  ,'12th','Doctorate' ,'5th-6th'  ,'1st-4th' ,'Preschool' ],y=data["education"].value_counts() ,
		marker=dict(color=['rgb(51, 153, 255)','rgb(255, 102, 102)','rgb(0,110,230)','rgb(0, 70, 150)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)'],opacity=0.75))]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Education',xref='paper',x=0),
	xaxis=dict(title="Education",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_barplot4(data):
	data = [go.Bar(x=['Prof-specialty','Craft-repair','Exec-managerial','Adm-clerical','Sales','Other-service','Machine-op-inspct','Transport-moving','Handlers-cleaners','Farming-fishing','Tech-support','Protective-serv','Priv-house-serv','Armed-Forces'],y=data["occupation"].value_counts(), opacity=0.75,
		marker=dict(color=['rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(51, 153, 255)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)','rgb(255, 102, 102)']))]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Occupation',xref='paper',x=0),
	xaxis=dict(title="Occupation",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_histplot4(data):
	data = [go.Histogram(x=data["capital-loss"],nbinsx=10,marker=dict(color='rgb(200, 0, 0)'),opacity=0.6 )]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Capital-Loss',xref='paper',x=0),
	xaxis=dict(title="Capital-Loss",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Count',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_barplot5(data):
	data = [go.Bar(x=['Male','Female'],y=data["gender"].value_counts(),marker=dict(color=['rgb(255, 102, 102)','rgb(51, 153, 255)'],opacity=0.6))]
	layout = go.Layout(autosize=False,width=450,height=450,title=go.layout.Title(text='Gender',xref='paper',x=0),
	xaxis=dict(title="Gender",titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black')))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

@app.route("/")
@app.route('/dashboard')
def plots():
    data = load_data()    
    hist1=create_histplot1(data)
    hist2=create_histplot2(data)
    bar1=create_barplot1(data)
    bar2=create_barplot2(data)
    hist3=create_histplot3(data)
    bar3=create_barplot3(data)
    bar4=create_barplot4(data)
    hist4=create_histplot4(data)
    bar5=create_barplot5(data)
    return render_template("dashboard.html",plot1=hist1,plot2=hist2,plot3=bar1,plot4=bar2, plot5=hist3, plot6=bar3,plot7=bar4,plot8=hist4,plot9=bar5)

@app.route('/data')
def data():
	return render_template("data.html")


@app.route('/form_prediction')
def form_prediction():
    return flask.render_template('form_prediction.html')


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("C:\\Users\\u23e00\\Downloads\\Income webapp\\Income-Prediction-Webapp-master\\WebApp\\model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/form_prediction2',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result)==1:
            prediction='Income more than 50K'
        else:
            prediction='Income less that 50K'
        return render_template("form_prediction.html",prediction=prediction)

if __name__ == '__main__':
	app.jinja_env.auto_reload=True
	app.config['TEMPLATES_AUTO_RELOAD']=True
	app.run(debug=True)
    








