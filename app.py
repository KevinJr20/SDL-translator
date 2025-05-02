from flask import Flask, request, render_template, jsonify
from translator import CulturalTranslator
import json

app = Flask(__name__)
translator = CulturalTranslator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.form
    phrase = data.get('phrase')
    lang = data.get('lang') or None
    context = data.get('context', 'casual')
    use_ai = data.get('use_ai') == 'on'
    reverse = data.get('reverse') == 'on'

    result = translator.translate(phrase, lang, reverse, use_ai, context)
    return jsonify(result)

@app.route('/history', methods=['GET'])
def history():
    return jsonify(translator.view_history())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    history = []
    return jsonify({'success': 'History cleared'})

@app.route('/add_phrase', methods=['POST'])
def add_phrase():
    data = request.form
    source = data.get('source')
    target = data.get('target')
    lang = data.get('lang')
    vibe = data.get('vibe')
    result = translator.add_phrase(source, target, lang, vibe)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)