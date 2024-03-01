from flask import Flask, request, flash, render_template
app = Flask(__name__)

#Rutas

@app.route('/')           # indica que dirección dispara la función
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/confusion_matrix')
def confusion_matrix():
    return render_template('confusion_matrix.html')

@app.route('/manual_metrics')
def manual_metrics():
    return render_template('metrics.html')

@app.route('/precision_curve')
def precision_curve():
    return render_template('precision_curve.html')

@app.route('/roc_curve')
def roc_curve():
    return render_template('roc_curve.html')

#Invalid URL
@app.errorhandler(404)
def page_not_found(e):
     return 'This page does not exist', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)