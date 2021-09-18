import flask
import json
from flask import request
from socket import socket
import cv2
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


app = flask.Flask(__name__)


@app.route("/members")
def members():
    return {"members": ["member1", "member2"]}


@app.route("/price", methods=['POST'])
def pricePrediction():


    price_type = ""
    district = ""
    vegitable = ""
    montht = 0
    datee = 0
    price_date = {
        "01": {
            # carrotRetail price
            "Retail": 1,
            "Vegitable": 1,
            "vals": [233.08600482011752, -21.0401426, 1004.87174, -0.777730696]
        },
        "02": {
            # carrotWholesale
            "Retail": 2,
            "Vegitable": 1,
            "vals": [113.58337507005217, 3.38147324, 1004.67816, -0.485063986]
        },
        "03": {
            # BeansRetail
            "Retail": 1,
            "Vegitable": 2,
            "vals": [198.1324482533537, -14.1908119, 1005.54437, 0.270519978]
        },
        "04": {
            # BeansWholesale
            "Retail": 2,
            "Vegitable": 2,
            "vals": [124.23611023889134, -2.84674655, 1005.06659, -0.175839571]
        },
        "05": {
            # coconutRetail
            "Retail": 1,
            "Vegitable": 3,
            "vals": [79.52501939001831, -1.70779719, 998.152261, -0.0992744985]
        },
        "06": {
            # coconutWholesale
            "Retail": 2,
            "Vegitable": 3,
            "vals": [68.93815979134888, -2.59949283, 998.748858, -0.00307293879]
        },

    }

    vals = []


    def get_vals(retail, district, vegitable):
        for x, y in price_date.items():
            if (y.get("Retail") == retail) & (y.get("Vegitable") == vegitable):
                vals.append(y.get("vals"))
                return vals

    data = request.get_json()

    user_type = int(data['user_type'])
    Dis_type = int(data['Dis_type'])
    vegi_type = int(data['vegi_type'])
    Month_type = int(data['Month_type'])
    date_type = int(data['date_type'])

    if ((user_type >= 1) & (user_type <= 2)) & ((Dis_type >= 1) & (Dis_type <= 3)) & ((vegi_type >= 1) & (vegi_type <= 3)) & ((Month_type >= 1) & (Month_type <= 12)) & ((date_type >= 1) & (date_type <= 31)):

        get_vals(user_type, Dis_type, vegi_type)

        x1 = float(vals[0][0])
        x2 = float(vals[0][1])
        x3 = float(vals[0][2])
        x4 = float(vals[0][3])

    price = ((x1)+(x2*Dis_type)+(x3*Month_type)+(x4*date_type))
    RealPrice = price - (Month_type*1000)

    return json.dumps({'result': 'success!',
                       'price': RealPrice})


@app.route("/yield", methods=['POST'])
def YieldPrediction():
    data = request.get_json()

    rainfall = float(data['rainfall'])
    temperature = float(data['temp'])
    wetdays = float(data['wetdays'])
    crop_count = float(data['crop_count'])
    User_Region = data['region']
    prediction_month= float(data['month'])


    if User_Region in "1":

        print("Region is Kurunegala")
        c = -91262.49928092473
        b1 = -374.17703859
        b2 = 747.97868378
        b3 = -2547.56918447
        b4 = 2987.08478943

        predicted_value = b1*prediction_month+b2*rainfall+b3*wetdays+b4*temperature+c
        predicted_value_new = predicted_value/13000
        monthly_yield = predicted_value_new*crop_count
        predicted_value_new = int(predicted_value_new)
        monthly_yield = int(monthly_yield)
        print("Monthly Yield Prediction per coconut tree:", predicted_value_new)
        print("Monthly Yield Prediction for your state:", monthly_yield)
        
        return json.dumps({'result':'',
                     'yield': predicted_value_new,
                     'yield1':monthly_yield})

    elif User_Region in "2":
        
        c = -72126.80256380427
        b1 = -441.22721113
        b2 = 788.66621641
        b3 = -2947.46919102
        b4 = 3239.97816901

        predicted_value = b1*prediction_month+b2*rainfall+b3*wetdays+b4*temperature+c
        predicted_value_new = predicted_value/13000
        monthly_yield = predicted_value_new*crop_count
        predicted_value_new = int(predicted_value_new)
        monthly_yield = int(monthly_yield)

        print("Monthly Yield Prediction per coconut tree:", predicted_value_new)
        print("Monthly Yield Prediction for your state:", monthly_yield)
        
        return json.dumps({'result':'',
                     'yield': predicted_value_new,
                     'yield1':monthly_yield})

    elif User_Region in "3":
        print("Region is Puttalam")
        c = -114552.90875936556
        b1 = -330.58720754
        b2 = 781.26749633
        b3 = -2965.6641107
        b4 = 2940.50105106

        predicted_value = b1*prediction_month+b2*rainfall+b3*wetdays+b4*temperature+c
        predicted_value_new = predicted_value/13000
        monthly_yield = predicted_value_new*crop_count
        predicted_value_new = int(predicted_value_new)
        monthly_yield = int(monthly_yield)

        print("Monthly Yield Prediction per coconut tree:", predicted_value_new)
        print("Monthly Yield Prediction for your state:", monthly_yield)
        

        return json.dumps({'result':'',
                     'yield': predicted_value_new,
                     'yield1': monthly_yield})





@app.route("/insect", methods=['POST'])
def Insectidentification():
    data = request.get_json()
    imagepath = data['path']
    imagename = data['imagename']

    CATEGORIES = ["Locusta_migratoria \n How to manage : Nymphs - ground spraying,Large nymph bands can be sprayed with boom sprays. Isolated and small areas can be sprayed using misting machines or knapsack sprayers.",
                  "Paddy_stem_maggot \n How to manage :The rice plant can compensate for the damage caused by the rice whorl maggot. Usually, the symptoms disappear during the maximum tillering stage of the crop."]

    def prepare(filepath):
        IMG_SIZE = 150
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    model = tf.keras.models.load_model("insects_category1-CNN.model")

    img = image.load_img(imagepath)
    plt.imshow(img)
    prediction = model.predict([prepare(imagename)])
    print(CATEGORIES[int(prediction[0][0])])

    return json.dumps({'prediction is': CATEGORIES[int(prediction[0][0])]})




if __name__ == '__main__':
    app.run(debug=True)
