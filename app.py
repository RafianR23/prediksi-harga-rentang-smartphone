from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler



model = pickle.load(open('finalized_model_new.pkl', 'rb'))
scaler = StandardScaler()

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')

# @app.route('/after_new')
# def after_new():
#     return render_template('after_new.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['daya_baterai']
    data2 = request.form['bluetooth']
    data3 = request.form['kecepatan_clock']
    data4 = request.form['dual_sim']
    data5 = request.form['kamera_depan']
    data6 = request.form['four_g']
    data7 = request.form['memori_internal']
    data8 = request.form['tebal_hp']
    data9 = request.form['berat_hp']
    data10 = request.form['jumlah_prosesor']
    data11 = request.form['kamera_belakang_mp']
    data12 = request.form['px_panjang']
    data13 = request.form['px_lebar']
    data14 = request.form['kapasitas_ram']
    data15 = request.form['panjang_layar']
    data16 = request.form['lebar_layar']
    data17 = request.form['waktu_telfon']
    data18 = request.form['three_g']
    data19 = request.form['touch_screen']
    data20 = request.form['wifi']
    arr = np.array([[float(data1), 
                     float(data2), 
                     float(data3), 
                     float(data4),
                     float(data5),
                     float(data6),
                     float(data7),
                     float(data8),
                     float(data9),
                     float(data10),
                     float(data11),
                     float(data12),
                     float(data13),
                     float(data14),
                     float(data15),
                     float(data16),
                     float(data17),
                     float(data18),
                     float(data19),
                     float(data20),
                     ]])
    new = scaler.fit_transform(arr)
    pred = model.predict(new)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















