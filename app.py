from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time
import pandas as pd  # Para construir window_df y pasar a heurísticas

# Importar la NUEVA función de features, nombres de columnas y heurísticas
from utils import (
    calculate_frame_features_v2,
    FEATURE_COLUMNS_V2,
    refine_activity_with_heuristics,
)

# -------------------------------------------------------
# Configuración general
# -------------------------------------------------------

app = Flask(__name__)

MODEL_PATH = "model/best_xgboost_v2.joblib"
SCALER_PATH = "model/scaler_v2.joblib"
LABEL_ENCODER_PATH = "model/label_encoder_v2.joblib"

DEBUG_PREDICTION = False  # ponlo en True si quieres ver prints de debug

# Cargar modelo, scaler y label encoder
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("Modelo v2, Scaler v2 y LabelEncoder v2 cargados correctamente.")
    print("Clases:", list(le.classes_))
except FileNotFoundError as e:
    print(f"Error cargando archivos del modelo: {e}")
    print(
        "Asegúrate de que 'best_xgboost_v2.joblib', 'scaler_v2.joblib' y "
        "'label_encoder_v2.joblib' estén en la carpeta 'model'."
    )
    exit()

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# Ventana deslizante (debe coincidir con el entrenamiento)
WINDOW_SIZE = 30
FEATURE_COUNT = len(FEATURE_COLUMNS_V2)  # 7
feature_buffer = deque(maxlen=WINDOW_SIZE)

# Historial de predicciones crudas del modelo (IDs numéricos)
PREDICTION_HISTORY_SIZE = 30            # ~1 segundo si vas ~30 FPS
prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

# Cada cuánto tiempo se actualiza el texto en pantalla
UPDATE_INTERVAL = 0.5                   # segundos
last_update_time = 0.0

current_prediction = "Initializing..."


# -------------------------------------------------------
# Rutas Flask
# -------------------------------------------------------

@app.route("/")
def index():
    """Página principal."""
    return render_template("index.html")


def generate_frames():
    """
    Genera frames para el stream de video.
    - Calcula features v2 por frame (7 features).
    - Arma ventanas de 30 frames.
    - Resume cada ventana en 35 features (estadísticas) como en el notebook.
    - Pasa al modelo, suaviza en el tiempo y aplica heurísticas geométricas.
    """
    global current_prediction, feature_buffer, prediction_history, last_update_time

    cap = cv2.VideoCapture(0)  # 0 = webcam por defecto

    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: No se pudo leer el frame de la webcam.")
            break

        # --------- MediaPipe: detección de pose ----------
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        frame_features = None

        if results.pose_landmarks:
            landmarks_current = results.pose_landmarks.landmark

            # NUEVA función de features (7 features, sin frame previo)
            frame_features = calculate_frame_features_v2(landmarks_current)

            # Dibujar landmarks para referencia
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        else:
            # Si perdemos la pose, reseteamos el buffer y las predicciones
            feature_buffer.clear()
            prediction_history.clear()
            current_prediction = "No Pose Detected"

        # --------- Lógica de predicción ----------
        if frame_features is not None:
            feature_buffer.append(frame_features)

            if len(feature_buffer) == WINDOW_SIZE:
                # Ventana completa → (30, 7)
                window_array = np.array(feature_buffer, dtype=np.float32)

                # Convertimos a DataFrame para replicar la lógica del notebook
                window_df = pd.DataFrame(
                    window_array,
                    columns=FEATURE_COLUMNS_V2,
                )

                # Mismas estadísticas que en create_windows_from_video:
                # mean, std, min, max, median por cada feature
                stats = window_df[FEATURE_COLUMNS_V2].agg(
                    ["mean", "std", "min", "max", "median"]
                ).transpose()

                # Vector final de 35 features: (7 features * 5 estadísticas)
                window_features = stats.values.flatten().reshape(1, -1)  # (1, 35)

                # Escalar
                scaled_data = scaler.transform(window_features)

                if DEBUG_PREDICTION:
                    print("\n--- Ventana completa ---")
                    print(f"Shape feature_buffer: {window_array.shape}")
                    print(
                        f"Shape window_features (antes de escalar): {window_features.shape}")
                    print(
                        f"Primeras 5 features escaladas: {scaled_data[0, :5]}")

                # Predecir y suavizar
                try:
                    prediction_encoded = int(model.predict(scaled_data)[0])

                    # Guardar predicción cruda en el historial
                    prediction_history.append(prediction_encoded)

                    # ¿Toca actualizar el texto visible?
                    now = time.time()
                    if (
                        now - last_update_time >= UPDATE_INTERVAL
                        and len(prediction_history) >= 5  # mínimo para votar
                    ):
                        values, counts = np.unique(
                            np.array(prediction_history), return_counts=True
                        )
                        dominant_encoded = int(values[np.argmax(counts)])

                        raw_label = le.inverse_transform([dominant_encoded])[0]

                        # Aplicar heurísticas geométricas para corregir:
                        # - sentado vs de pie
                        # - transiciones sentarse / levantarse
                        # - walking vs standing_still
                        refined_label = refine_activity_with_heuristics(
                            window_df, raw_label
                        )

                        if DEBUG_PREDICTION:
                            print(
                                f"Raw label: {raw_label} -> Refined: {refined_label}")

                        current_prediction = refined_label
                        last_update_time = now

                    # Si aún no se cumple el intervalo, dejamos current_prediction como está

                except Exception as e:
                    current_prediction = f"Prediction Error: {e}"
            else:
                # Todavía no hay ventana completa
                current_prediction = (
                    f"Collecting frames... "
                    f"({len(feature_buffer)}/{WINDOW_SIZE})"
                )

        # --------- Dibujar la predicción en el frame ----------
        COLOR_MAP = {
            "sitting_down": (0, 255, 255),               # amarillo
            "sitting_still": (255, 0, 255),              # magenta
            "standing_still": (0, 255, 0),               # verde
            "standing_up": (255, 255, 0),                # cian
            "turning": (255, 0, 0),                      # azul
            "walking_away_from_camera": (0, 165, 255),   # naranja
            "walking_towards_camera": (0, 0, 255),       # rojo
        }

        default_color = (255, 255, 255)
        color = COLOR_MAP.get(current_prediction, default_color)

        cv2.putText(
            image_bgr,
            f"Activity: {current_prediction}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        # --------- Codificar y devolver frame ----------
        try:
            ret, buffer = cv2.imencode(".jpg", image_bgr)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        except Exception as e:
            print(f"Error codificando frame: {e}")
            continue

    cap.release()
    print("Webcam liberada.")


@app.route("/video_feed")
def video_feed():
    """Ruta de streaming de video."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# -------------------------------------------------------
# Lanzar la app
# -------------------------------------------------------

if __name__ == "__main__":
    # host='0.0.0.0' → accesible en la red local
    app.run(debug=True, host="0.0.0.0", port=5000)
