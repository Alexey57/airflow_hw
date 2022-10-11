import pandas as pd
import json
import requests
import dill
import os
import sys
from datetime import datetime


# path = ('C:/Users/папа/airflow_hw')   # текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '..')  # при запуске в Airflow


def predict():

    pkl_names = os.listdir(f'{path}/data/models')[-1]
    with open(f'{path}/data/models/'+pkl_names, 'rb') as file:
        model = dill.load(file)
    dir_list = os.listdir(f'{path}/data/test')
    data = []
    for i in range(len(dir_list)):
        with open(f'{path}/data/test/' + dir_list[i], 'r') as j:
            contents = json.load(j)
        data.append(contents)
    df = pd.DataFrame.from_dict(data)
    y = model.predict(df)
    df['predict'] = y
    df_pred = df[['id', 'predict']]
    predict_failname = f'{path}/data/predictions/car_price_prediction_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_pred.to_csv(predict_failname)


if __name__ == '__main__':
    predict()


