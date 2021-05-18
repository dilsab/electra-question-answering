import flask.scaffold
from flask import Flask, render_template

from answering import ElectraQuestionAnswering

flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restful import reqparse, Api, Resource

electra_question_answering = ElectraQuestionAnswering('save')
app = Flask(__name__, template_folder='templates')
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('context', type=str)
parser.add_argument('question', type=str)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


class Answer(Resource):
    def get(self):
        args = parser.parse_args()

        if any([value is None for value in args.values()]):
            return {'error': 'bad_request', 'code': 400}, 400

        answer = electra_question_answering.predict_answer(args['context'], args['question'])

        return {'answer': answer}


api.add_resource(Answer, '/answer')

if __name__ == '__main__':
    app.run(debug=True)
