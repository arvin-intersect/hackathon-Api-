# 🌟 Flow Habits AI: Digital Well-being Prediction API 🌟

![Status](https://img.shields.io/badge/Status-Hackathon%20Project-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)

## 📂 Directory Structure

```
arvin-intersect-hackathon-api-/
├── api.py                    # The core Flask API application
├── data.py                   # Script to generate synthetic dataset and metadata
├── model_config.json         # Configuration and metadata for the ML model
├── model_statistics.json     # Detailed statistics and metrics of the trained model
├── requirements.txt          # Python dependencies for the project
└── tuning.py                 # Script for training, evaluating, and saving the ML model
```

## 🚀 Getting Started

Follow these steps to set up and run the Flow Habits AI backend API locally.

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** (or newer)
- **pip** (Python package installer)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/arvin-intersect-hackathon-api.git
   cd arvin-intersect-hackathon-api
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Generating Data & Training the Model

Before running the API, you need to generate the synthetic dataset and train the machine learning model. This ensures `data.csv` and `stress_model.pkl` are available.

1. **Generate Synthetic Data:**
   
   This script (`data.py`) creates a synthetic dataset (`stress_prediction_dataset_90percent.csv`) and its metadata (`dataset_metadata_90percent.json`), which is used for model training.
   
   ```bash
   python data.py
   ```

2. **Train and Save the ML Model:**
   
   This script (`tuning.py`) trains a RandomForestClassifier using the generated data, evaluates its performance, saves the trained model as `stress_model.pkl`, and updates `model_config.json` and `model_statistics.json` with the latest metrics. It also generates several visualizations in the `visualizations/` directory.
   
   ```bash
   python tuning.py
   ```
   
   *(Note: You might see a warning about data.csv not existing if you run tuning.py before data.py. Ensure data.py is run first.)*

### Running the API

Once the model is trained and saved, you can start the Flask API server:

```bash
python api.py
```

The API will start running on `http://0.0.0.0:5000` (or `http://127.0.0.1:5000`).

## 🌐 API Endpoints

The Flow Habits AI API provides the following endpoints:

### 1. Root Endpoint (`/`)

A basic endpoint to check if the API is running and get general information.

- **Method**: `GET`
- **URL**: `http://localhost:5000/`
- **Response:**
  ```json
  {
      "api": "Stress Prediction API",
      "version": "1.0",
      "status": "active",
      "model_accuracy": "99.75%",
      "endpoints": {
          "/predict": "POST - Predict stress level",
          "/correlations": "GET - Feature correlations",
          "/stats": "GET - Model statistics",
          "/health": "GET - API health"
      },
      "features_required": [
          "Technology_Usage_Hours",
          "Social_Media_Usage_Hours",
          "Gaming_Hours",
          "Screen_Time_Hours",
          "Sleep_Hours",
          "Physical_Activity_Hours"
      ],
      "classes": ["0", "1", "2"]
  }
  ```

### 2. Health Check (`/health`)

Checks the health status of the API and model loading.

- **Method**: `GET`
- **URL**: `http://localhost:5000/health`
- **Response:**
  ```json
  {
      "status": "healthy",
      "timestamp": "2023-10-27T10:30:00.123456",
      "model_loaded": true
  }
  ```

### 3. Feature Correlations (`/correlations`)

Retrieves the correlation of input features with stress levels.

- **Method**: `GET`
- **URL**: `http://localhost:5000/correlations`
- **Response:**
  ```json
  {
      "correlations": {
          "Physical Activity Hours": -0.837830889269115,
          "Gaming Hours": 0.8457583584204861
      },
      "interpretation": {
          "positive": "Higher value = More stress",
          "negative": "Higher value = Less stress"
      },
      "top_stress_factors": [
          { "feature": "Screen Time Hours", "correlation": 0.9402817457510879 }
      ],
      "top_stress_reducers": [
          { "feature": "Sleep Hours", "correlation": -0.8687572277054169 }
      ]
  }
  ```

### 4. Model Statistics (`/stats`)

Provides detailed statistics and configuration of the trained machine learning model.

- **Method**: `GET`
- **URL**: `http://localhost:5000/stats`
- **Response**: Returns the content of `model_statistics.json`.

### 5. Predict Stress Level (`/predict`)

The core prediction endpoint. It accepts user input features, predicts the stress level, and provides personalized recommendations.

- **Method**: `POST`
- **URL**: `http://localhost:5000/predict`
- **Request Body (JSON):**
  ```json
  {
      "technology_hours": 8.5,
      "social_media_hours": 4.2,
      "gaming_hours": 1.5,
      "screen_time_hours": 14.2,
      "sleep_hours": 5.8,
      "physical_activity_hours": 0.5
  }
  ```
  *(All fields are required.)*

- **Response (Example for "High" stress):**
  ```json
  {
      "prediction": {
          "stress_level": "High",
          "confidence": 0.957,
          "probabilities": {
              "0": 0.012,
              "1": 0.031,
              "2": 0.957
          }
      },
      "input_data": {
          "Technology_Usage_Hours": 8.5,
          "Social_Media_Usage_Hours": 4.2,
          "Gaming_Hours": 1.5,
          "Screen_Time_Hours": 14.2,
          "Sleep_Hours": 5.8,
          "Physical_Activity_Hours": 0.5
      },
      "recommendations": {
          "stress_level": "High",
          "message": "🔴 Your stress level is HIGH. Take action now!",
          "tasks": [
              {
                  "task": "🌳 Go outside",
                  "points": 50,
                  "duration": "30 min",
                  "description": "Fresh air reduces cortisol"
              },
              {
                  "task": "📚 Read a book",
                  "points": 40,
                  "duration": "20 min",
                  "description": "Give your eyes a break"
              }
          ],
          "insights": [
              {
                  "type": "warning",
                  "message": "Screen time very high (14.2h)",
                  "action": "Reduce 3h today"
              },
              {
                  "type": "critical",
                  "message": "Sleep deprived (5.8h)",
                  "action": "Aim for 8h tonight"
              }
          ],
          "gamification": {
              "current_level": "Stress Warrior",
              "points_needed": 150,
              "next_level": "Zen Master",
              "challenge": "Complete 3 tasks",
              "streak": 0,
              "badge": "🔥 Stress Fighter"
          }
      },
      "timestamp": "2023-10-27T10:30:00.123456"
  }
  ```

## 📊 Model Insights & Visualizations

The `tuning.py` script not only trains the model but also generates a set of insightful visualizations saved in the `visualizations/` directory. These graphics provide a deeper understanding of the model's performance and the underlying data patterns:

- **confusion_matrix.png**: Visualizes the model's classification accuracy across stress levels.
- **feature_importance.png**: Highlights which features (e.g., Screen Time, Sleep Hours) contribute most to stress prediction.
- **correlation_heatmap.png**: Shows the relationships between all input features and stress levels.
- **correlation_bars.png**: Clearly displays the positive/negative correlation of each feature with stress.
- **stress_distribution.png**: Illustrates the distribution of stress levels in the training data.
- **feature_vs_stress.png**: Scatter plots showing how individual features correlate with stress levels.
- **classification_report.png**: A heatmap summary of precision, recall, and F1-score for each stress class.

These visualizations are key to understanding the model's excellent performance and the impactful factors influencing digital well-being.

## 📈 Future Enhancements

As a hackathon project, Flow Habits AI has immense potential for growth:

- **Mobile App Integration**: Direct integration with mobile platforms to automatically collect screen time data (e.g., from Apple Health, Google Digital Wellbeing) and deliver recommendations.
- **Wearable Device Support**: Connect with smartwatches (e.g., Apple Watch) for richer activity and sleep data.
- **User Feedback Loop**: Implement mechanisms for users to provide feedback on recommendations, further personalizing the model.
- **Advanced Gamification**: Expand gamified elements with leaderboards, peer challenges, and more diverse reward structures.
- **Dynamic Task Generation**: Integrate a more sophisticated task generation engine that considers user preferences, time of day, and external factors.
- **Explainable AI (XAI)**: Provide more transparent explanations for predictions and recommendations.


---

*Made with ❤️ for better digital well-being*

## 🚀 Project Overview

Welcome to the backend of **Flow Habits AI**, an innovative solution designed to empower individuals to master their digital well-being. Developed for the **BITS Hackathon** by **Arvin Subramanian,Demont Fort University Team - Agentic Force**, this API leverages machine learning to predict user stress levels based on digital habits and provides hyper-personalized, gamified recommendations to foster healthier digital behaviors.

In an era where the average individual spends increasing hours on screens, leading to elevated stress and decreased productivity, Flow Habits AI offers a data-driven approach. It analyzes usage patterns (like screen time, social media, sleep, and physical activity) to not only identify potential stress but also guide users towards mindfulness and balance through actionable tasks, insights, and engaging challenges.

**Theme:** Health Tech and Wellbeing

## ✨ Key Features

- **Stress Level Prediction**: Predicts a user's current stress level (Low, Medium, High) with high accuracy (approx. 99.75%).
- **Hyper-Personalized Recommendations**: Delivers tailored tasks, insights, and gamified elements based on the predicted stress level and specific user data.
- **Gamification Engine**: Includes points, levels, challenges, streaks, and badges to motivate sustained engagement and habit formation.
- **Real-time Insights**: Provides immediate feedback on screen time, sleep, and physical activity, highlighting critical or warning patterns.
- **Dynamic API Endpoints**: A robust Flask API offering prediction services, model statistics, feature correlations, and health checks.
- **Synthetic Data Generation**: Includes a script to generate a realistic dataset simulating digital habits and stress levels for robust model training.
- **Comprehensive Model Training & Evaluation**: A dedicated script for training a RandomForestClassifier, evaluating its performance, and generating insightful visualizations (e.g., confusion matrix, feature importance, correlations).
- **Configurable & Scalable**: Model configuration and statistics are externalized in JSON files, making the system easy to manage and integrate.

## 💡 How It Works (The Flow)

Flow Habits AI operates through a sophisticated, yet intuitive, flow:

1. **Data Collection (Frontend Input)**: Users provide their digital usage and lifestyle data (e.g., screen time, sleep hours, physical activity). For the hackathon, this is simulated via API input, with future plans for direct app integration (e.g., Apple Health, Google Fit).
2. **Stress Prediction (API)**: The input data is fed into our highly accurate machine learning model via the `/predict` API endpoint.
3. **Personalized Intervention (API)**: Based on the predicted stress level, the API's recommendation engine crafts a unique set of tasks, insights, and gamified challenges.
4. **User Engagement (Frontend Display)**: These recommendations are delivered to the user, motivating healthier digital behaviors and tracking their progress through points and badges.

## 🛠️ Tech Stack

- **Backend Framework**: Flask
- **Machine Learning**: Scikit-learn (RandomForestClassifier)
- **Data Manipulation**: Pandas, NumPy
- **Serialization**: Joblib, JSON
- **API Utilities**: Flask-CORS
- **Deployment**: Docker (recommended for production, not explicitly in files but implied by 0.0.0.0 host)
- **Programming Language**: Python 3.9+

All the code components are publicly available at https://github.com/arvin-intersect/hackathon-Api-


