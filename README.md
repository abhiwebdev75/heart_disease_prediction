# 💖 CardioPredict AI: Heart Disease Predictor

## ⚡ Project Overview

CardioPredict AI is a lightweight, high-contrast web application designed for the instant input and prediction of heart disease risk based on standard patient clinical features.

This project demonstrates a basic web application architecture integrating a pre-trained Machine Learning model for inference, featuring a sleek dark-mode UI with "glassmorphism" effects and privacy-focused client-side storage for reports.

---

## ✨ Key Features

* **13 Clinical Features:** Comprehensive data input covering patient profile, vitals, and stress diagnostics (based on the standard Cleveland Heart Disease dataset).
* **Flask Backend:** Uses a simple Python backend with Flask to serve the application and manage model prediction requests.
* **ML Integration:** Loads and utilizes pre-trained `Scikit-learn` model (`.pkl` file) and `StandardScaler` for production-ready inference.
* **Data Privacy:** Uses **`localStorage`** in the browser to save and load patient reports, ensuring data never leaves the client machine.
* **Interactive UI:** Features custom input styling, vibrant, defined buttons, and a dynamic heartbeat background that disappears upon submission.

---

## 🛠️ Technology Stack

| Area | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3, **Flask** | Server, API routing (`/predict`), environment setup. |
| **Model/Data** | NumPy, **Scikit-learn**, `joblib` | Model loading, feature scaling, and prediction. |
| **Frontend** | HTML5, CSS3, **Tailwind CSS (CDN)** | Structure and utility-first styling. |
| **Interactivity** | Vanilla JavaScript (ESM), Lucide Icons | Form validation, UI state management, localStorage logic. |

---

## 🚀 Setup and Installation

Follow these steps to get your local copy of CardioPredict AI running.

### Prerequisites

1.  Python 3.x and `pip`
2.  Pre-trained model files: `heart_disease_prediction_model.pkl` and `scaler.pkl`

### Step 1: Clone the Repository

```bash
git clone heart_disease_prediction
cd CardioPredict-AI
