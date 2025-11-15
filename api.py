from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

model = joblib.load('stress_model.pkl')

with open('model_config.json', 'r') as f:
    config = json.load(f)

with open('model_statistics.json', 'r') as f:
    stats = json.load(f)

if 'classes' not in config:
    config['classes'] = ['0', '1', '2']

def get_recommendations(stress_level, user_data):
    recommendations = {
        'stress_level': stress_level,
        'message': '',
        'tasks': [],
        'insights': [],
        'gamification': {}
    }
    if stress_level == 'High':
        recommendations['message'] = '🔴 Your stress level is HIGH. Take action now!'
        recommendations['tasks'] = [
            {'task': '🌳 Go outside', 'points': 50, 'duration': '30 min', 'description': 'Fresh air reduces cortisol'},
            {'task': '📚 Read a book', 'points': 40, 'duration': '20 min', 'description': 'Give your eyes a break'},
            {'task': '🧘 Meditation', 'points': 30, 'duration': '10 min', 'description': 'Calm your nervous system'},
            {'task': '🏃 Physical exercise', 'points': 60, 'duration': '30 min', 'description': 'Release endorphins'},
            {'task': '😴 Power nap', 'points': 35, 'duration': '20 min', 'description': 'Recharge your energy'}
        ]
        if user_data['Screen_Time_Hours'] > 12:
            recommendations['insights'].append({'type':'warning','message':f'Screen time very high ({user_data["Screen_Time_Hours"]:.1f}h)','action':'Reduce 3h today'})
        if user_data['Sleep_Hours'] < 6:
            recommendations['insights'].append({'type':'critical','message':f'Sleep deprived ({user_data["Sleep_Hours"]:.1f}h)','action':'Aim for 8h tonight'})
        if user_data['Physical_Activity_Hours'] < 1:
            recommendations['insights'].append({'type':'warning','message':'Almost no physical activity','action':'Add 30 min movement'})
        recommendations['gamification'] = {'current_level':'Stress Warrior','points_needed':150,'next_level':'Zen Master','challenge':'Complete 3 tasks','streak':0,'badge':'🔥 Stress Fighter'}
    elif stress_level == 'Medium':
        recommendations['message'] = '🟡 Your stress level is MODERATE. Let\'s improve it!'
        recommendations['tasks'] = [
            {'task': '🚶 Take a walk', 'points':30,'duration':'15 min','description':'Light movement improves mood'},
            {'task': '💧 Drink water', 'points':10,'duration':'5 min','description':'Stay hydrated'},
            {'task': '📱 Reduce social media', 'points':25,'duration':'Throughout day','description':'Less scrolling = less stress'},
            {'task': '🎵 Listen to music', 'points':20,'duration':'15 min','description':'Music therapy reduces anxiety'},
            {'task': '🌅 Morning sunlight', 'points':35,'duration':'30 min','description':'Regulates circadian rhythm'}
        ]
        if user_data['Screen_Time_Hours'] > 10:
            recommendations['insights'].append({'type':'info','message':f'Screen time elevated ({user_data["Screen_Time_Hours"]:.1f}h)','action':'Reduce 1-2h'})
        if user_data['Sleep_Hours'] < 7:
            recommendations['insights'].append({'type':'info','message':f'Sleep could be better ({user_data["Sleep_Hours"]:.1f}h)','action':'Aim 7-8h'})
        recommendations['gamification'] = {'current_level':'Balanced Soul','points_needed':100,'next_level':'Stress Warrior','challenge':'Complete 2 tasks','streak':0,'badge':'⚖️ Harmony Seeker'}
    else:
        recommendations['message'] = '🟢 Your stress level is LOW. Great job!'
        recommendations['tasks'] = [
            {'task':'🎉 Keep up the work','points':20,'duration':'Ongoing','description':'Habits are working'},
            {'task':'📝 Journal success habits','points':25,'duration':'10 min','description':'Document what\'s working'},
            {'task':'🤝 Help someone','points':40,'duration':'30 min','description':'Share strategies'},
            {'task':'🎯 Set new goal','points':30,'duration':'15 min','description':'Keep improving'}
        ]
        recommendations['insights'].append({'type':'success','message':'✅ Excellent! Your habits work','action':'Keep it up'})
        recommendations['insights'].append({'type':'success','message':f'🌟 Perfect balance: {user_data["Sleep_Hours"]:.1f}h sleep, {user_data["Screen_Time_Hours"]:.1f}h screen','action':'Maintain'})
        recommendations['gamification'] = {'current_level':'Zen Master','points_needed':50,'next_level':'Wellness Guru','challenge':'Maintain low stress 7 days','streak':0,'badge':'🏆 Stress Champion'}
    return recommendations

@app.route('/')
def home():
    return jsonify({
        'api': 'Stress Prediction API',
        'version': '1.0',
        'status': 'active',
        'model_accuracy': f"{stats.get('accuracy', 0)*100:.2f}%",
        'endpoints': {
            '/predict': 'POST - Predict stress level',
            '/correlations': 'GET - Feature correlations',
            '/stats': 'GET - Model statistics',
            '/health': 'GET - API health'
        },
        'features_required': config['features'],
        'classes': config['classes']
    })

@app.route('/health')
def health():
    return jsonify({'status':'healthy','timestamp': datetime.now().isoformat(),'model_loaded': True})

@app.route('/correlations')
def get_correlations_endpoint():
    return jsonify({
        'correlations': stats['correlations'],
        'interpretation': {'positive':'Higher value = More stress','negative':'Higher value = Less stress'},
        'top_stress_factors':[{'feature':k,'correlation':v} for k,v in sorted(stats['correlations'].items(), key=lambda x:x[1], reverse=True)[:3]],
        'top_stress_reducers':[{'feature':k,'correlation':v} for k,v in sorted(stats['correlations'].items(), key=lambda x:x[1])[:3]]
    })

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error':'No data provided'}), 400
        features_input = {
            'Technology_Usage_Hours': data.get('technology_hours'),
            'Social_Media_Usage_Hours': data.get('social_media_hours'),
            'Gaming_Hours': data.get('gaming_hours'),
            'Screen_Time_Hours': data.get('screen_time_hours'),
            'Sleep_Hours': data.get('sleep_hours'),
            'Physical_Activity_Hours': data.get('physical_activity_hours')
        }
        missing = [k for k,v in features_input.items() if v is None]
        if missing:
            return jsonify({'error':'Missing required features','missing':missing}),400
        input_df = pd.DataFrame([features_input])
        prediction_raw = model.predict(input_df)[0]
        prediction = int(prediction_raw)
        prediction_proba = model.predict_proba(input_df)[0]
        proba_dict = {str(cls): float(prob) for cls, prob in zip(config['classes'], prediction_proba)}
        stress_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        stress_label = stress_mapping.get(prediction, 'Unknown')
        recommendations = get_recommendations(stress_label, features_input)
        return jsonify({
            'prediction': {
                'stress_level': stress_label,
                'confidence': float(max(prediction_proba)),
                'probabilities': proba_dict
            },
            'input_data': features_input,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error':str(e),'type':type(e).__name__}),500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
