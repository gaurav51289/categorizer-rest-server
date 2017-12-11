from flask_api import FlaskAPI
from flask import request, jsonify
from flask_cors import CORS
from categorizer import getCategories
from predict_classifier import magpie

def create_app():
    app = FlaskAPI(__name__, instance_relative_config=True)

    CORS(app)

    @app.route('/categorize/', methods=['POST'])
    def ask():
        if request.method == "POST":
            question = str(request.data.get('question'))
            if question:

                que_cats = getCategories(question)

                response = jsonify({
                    'categories': que_cats
                })
                response.status_code = 200
                return response

    @app.route('/magcategorize/', methods=['POST'])
    def mag_ask():
        if request.method == "POST":
            question = str(request.data.get('question'))
            if question:

                que_cats = magpie.predict_from_text(question)
                print(que_cats)
                response = jsonify({
                    'categories': que_cats
                })
                response.status_code = 200
    return app
