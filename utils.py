import numpy as np

# ============================================================
# Nombres de landmarks según MediaPipe Pose
# (el orden DEBE coincidir con mp.solutions.pose.PoseLandmark)
# ============================================================

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Landmarks relevantes para la versión 1 (con velocidades)
RELEVANT_LANDMARK_INDICES = {
    name: LANDMARK_NAMES.index(name)
    for name in [
        "nose",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_wrist", "right_wrist",
    ]
}

# ------------------------------------------------------------
# FEATURES V1 (modelo viejo, 16 features con velocidades)
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "right_knee_angle", "left_knee_angle",
    "right_hip_angle", "left_hip_angle",
    "trunk_inclination",
    "vel_nose",
    "vel_left_shoulder", "vel_right_shoulder",
    "vel_left_hip", "vel_right_hip",
    "vel_left_knee", "vel_right_knee",
    "vel_left_ankle", "vel_right_ankle",
    "vel_left_wrist", "vel_right_wrist",
]

# ------------------------------------------------------------
# FEATURES V2 (modelo nuevo, 7 features simples)
# ------------------------------------------------------------

FEATURE_COLUMNS_V2 = [
    "normalized_leg_length",
    "shoulder_vector_x",
    "shoulder_vector_z",
    "ankle_vector_x",
    "ankle_vector_z",
    "average_hip_angle",
    "average_knee_angle",
]

# Si quieres ver prints de debug, pon esto en True
DEBUG_FEATURES = False


# ============================================================
# Función auxiliar: ángulo entre 3 puntos
# ============================================================

def calculate_angle(p1, p2, p3):
    """
    Calcula el ángulo en p2 formado por los vectores p1-p2 y p3-p2 (en grados).

    p1, p2, p3: iterables tipo [x, y, z] o np.array
    """
    v1 = np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
    v2 = np.array(p3, dtype=np.float32) - np.array(p2, dtype=np.float32)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return np.nan  # evita división por cero

    dot_product = float(np.dot(v1, v2))

    # Clamp para evitar cosenos ligeramente fuera de [-1, 1] por errores numéricos
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


# ============================================================
# VERSIÓN 1: calculate_frame_features (16 features + velocidades)
# ============================================================

def calculate_frame_features(landmarks, prev_landmarks=None):
    """
    Calcula las 16 características cinemáticas para un frame dado (versión antigua).

    Args:
        landmarks: lista de landmarks de MediaPipe para el frame actual
                   (results.pose_landmarks.landmark).
        prev_landmarks: lista de landmarks para el frame anterior
                        (para calcular velocidades). Puede ser None.

    Returns:
        np.array de shape (16,) con las features, o None si algo crítico falla.
    """
    if landmarks is None:
        return None

    coords = {}
    prev_coords = {}
    landmark_map = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}

    # -----------------------------
    # 1. Coordenadas normalizadas
    # -----------------------------
    try:
        left_hip = landmarks[landmark_map["left_hip"]]
        right_hip = landmarks[landmark_map["right_hip"]]

        if left_hip.visibility < 0.5 or right_hip.visibility < 0.5:
            if DEBUG_FEATURES:
                print("Advertencia: Caderas no detectadas con suficiente confianza.")
            return None

        hip_center_x = (left_hip.x + right_hip.x) / 2.0
        hip_center_y = (left_hip.y + right_hip.y) / 2.0
        hip_center_z = (left_hip.z + right_hip.z) / 2.0

        # Coordenadas del frame actual, centradas en la cadera
        for name, index in RELEVANT_LANDMARK_INDICES.items():
            lm = landmarks[index]
            coords[name] = [
                lm.x - hip_center_x,
                lm.y - hip_center_y,
                lm.z - hip_center_z,
            ]

        # Frame anterior (para velocidades)
        if prev_landmarks is not None:
            prev_left_hip = prev_landmarks[landmark_map["left_hip"]]
            prev_right_hip = prev_landmarks[landmark_map["right_hip"]]

            if prev_left_hip.visibility < 0.5 or prev_right_hip.visibility < 0.5:
                # No confiamos en el frame anterior
                prev_landmarks = None
                prev_coords = {}
            else:
                prev_hip_center_x = (prev_left_hip.x + prev_right_hip.x) / 2.0
                prev_hip_center_y = (prev_left_hip.y + prev_right_hip.y) / 2.0
                prev_hip_center_z = (prev_left_hip.z + prev_right_hip.z) / 2.0

                for name, index in RELEVANT_LANDMARK_INDICES.items():
                    prev_lm = prev_landmarks[index]
                    prev_coords[name] = [
                        prev_lm.x - prev_hip_center_x,
                        prev_lm.y - prev_hip_center_y,
                        prev_lm.z - prev_hip_center_z,
                    ]

    except (IndexError, AttributeError, KeyError) as e:
        if DEBUG_FEATURES:
            print(
                f"Advertencia: Error al acceder a landmarks necesarios - {e}")
        return None

    # -----------------------------
    # 2. Ángulos de rodilla y cadera
    # -----------------------------
    try:
        right_knee_angle = calculate_angle(
            coords["right_hip"], coords["right_knee"], coords["right_ankle"]
        )
        left_knee_angle = calculate_angle(
            coords["left_hip"], coords["left_knee"], coords["left_ankle"]
        )
        right_hip_angle = calculate_angle(
            coords["right_shoulder"], coords["right_hip"], coords["right_knee"]
        )
        left_hip_angle = calculate_angle(
            coords["left_shoulder"], coords["left_hip"], coords["left_knee"]
        )
    except KeyError as e:
        if DEBUG_FEATURES:
            print(
                f"Advertencia: Faltan coordenadas para calcular ángulos - {e}")
        return None

    # -----------------------------
    # 3. Inclinación del tronco
    # -----------------------------
    try:
        shoulder_mid = np.array(
            [
                (coords["left_shoulder"][0] +
                 coords["right_shoulder"][0]) / 2.0,
                (coords["left_shoulder"][1] +
                 coords["right_shoulder"][1]) / 2.0,
                (coords["left_shoulder"][2] +
                 coords["right_shoulder"][2]) / 2.0,
            ],
            dtype=np.float32,
        )
        # origen tras normalizar
        hip_mid = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        trunk_vector = shoulder_mid - hip_mid
        norm_trunk = np.linalg.norm(trunk_vector)

        trunk_inclination = 90.0  # valor por defecto

        if norm_trunk > 1e-6:
            # Ángulo con el vector vertical hacia ARRIBA [0, -1, 0]
            vertical_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            cos_angle_up = np.clip(
                np.dot(trunk_vector, vertical_up) / norm_trunk, -1.0, 1.0
            )
            angle_with_up_deg = np.degrees(np.arccos(cos_angle_up))
            # Interpretación: de pie ≈ 0, inclinado ≈ 90
            trunk_inclination = float(angle_with_up_deg)

    except KeyError as e:
        if DEBUG_FEATURES:
            print(
                f"Advertencia: Faltan coordenadas para calcular inclinación del tronco - {e}"
            )
        trunk_inclination = 90.0

    # -----------------------------
    # 4. Velocidades de landmarks
    # -----------------------------
    velocities = {}
    if prev_coords:
        for name in RELEVANT_LANDMARK_INDICES.keys():
            try:
                distance = np.linalg.norm(
                    np.array(coords[name], dtype=np.float32)
                    - np.array(prev_coords[name], dtype=np.float32)
                )
                velocities[f"vel_{name}"] = float(distance)
            except KeyError:
                velocities[f"vel_{name}"] = 0.0
    else:
        for name in RELEVANT_LANDMARK_INDICES.keys():
            velocities[f"vel_{name}"] = 0.0

    # -----------------------------
    # 5. Ensamblar vector de features
    # -----------------------------
    feature_vector = [
        right_knee_angle,
        left_knee_angle,
        right_hip_angle,
        left_hip_angle,
        trunk_inclination,
        velocities.get("vel_nose", 0.0),
        velocities.get("vel_left_shoulder", 0.0),
        velocities.get("vel_right_shoulder", 0.0),
        velocities.get("vel_left_hip", 0.0),
        velocities.get("vel_right_hip", 0.0),
        velocities.get("vel_left_knee", 0.0),
        velocities.get("vel_right_knee", 0.0),
        velocities.get("vel_left_ankle", 0.0),
        velocities.get("vel_right_ankle", 0.0),
        velocities.get("vel_left_wrist", 0.0),
        velocities.get("vel_right_wrist", 0.0),
    ]

    feature_vector = np.nan_to_num(
        np.array(feature_vector, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if feature_vector.shape[0] != len(FEATURE_COLUMNS):
        if DEBUG_FEATURES:
            print(
                f"Error: Feature vector length mismatch! "
                f"Esperado {len(FEATURE_COLUMNS)}, got {feature_vector.shape[0]}"
            )
        return None

    return feature_vector


# ============================================================
# VERSIÓN 2: calculate_frame_features_v2 (7 features para modelo nuevo)
# ============================================================

def calculate_frame_features_v2(landmarks):
    """
    Calcula las 7 características usadas por el modelo v2 (sin usar frame previo).

    Features:
        - normalized_leg_length
        - shoulder_vector_x, shoulder_vector_z
        - ankle_vector_x, ankle_vector_z
        - average_hip_angle
        - average_knee_angle

    Args:
        landmarks: lista de landmarks de MediaPipe para el frame actual
                   (results.pose_landmarks.landmark).

    Returns:
        np.array de shape (7,), o None si algún landmark clave falla.
    """
    if landmarks is None:
        return None

    landmark_map = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}

    def get_lm(name):
        idx = landmark_map[name]
        return landmarks[idx]

    try:
        # Landmarks imprescindibles
        required_names = [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]

        for name in required_names:
            lm = get_lm(name)
            if lm.visibility < 0.5:
                return None

        def to_arr(lm):
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        ls = to_arr(get_lm("left_shoulder"))
        rs = to_arr(get_lm("right_shoulder"))
        lh = to_arr(get_lm("left_hip"))
        rh = to_arr(get_lm("right_hip"))
        lk = to_arr(get_lm("left_knee"))
        rk = to_arr(get_lm("right_knee"))
        la = to_arr(get_lm("left_ankle"))
        ra = to_arr(get_lm("right_ankle"))

        hip_center = 0.5 * (lh + rh)
        shoulder_center = 0.5 * (ls + rs)
        ankle_center = 0.5 * (la + ra)

        # a) normalized_leg_length = |cadera→tobillos| / |cadera→hombros|
        leg_vec = ankle_center - hip_center
        leg_len = float(np.linalg.norm(leg_vec))

        torso_vec = shoulder_center - hip_center
        torso_len = float(np.linalg.norm(torso_vec))

        if torso_len <= 1e-6:
            normalized_leg_length = 0.0
        else:
            normalized_leg_length = leg_len / torso_len

        # b) shoulder_vector_x, shoulder_vector_z
        shoulder_vec = rs - ls
        sv_norm = float(np.linalg.norm(shoulder_vec))
        if sv_norm <= 1e-6:
            shoulder_unit = np.zeros(3, dtype=np.float32)
        else:
            shoulder_unit = shoulder_vec / sv_norm

        shoulder_vector_x = float(shoulder_unit[0])
        shoulder_vector_z = float(shoulder_unit[2])

        # c) ankle_vector_x, ankle_vector_z
        ankle_vec = ra - la
        av_norm = float(np.linalg.norm(ankle_vec))
        if av_norm <= 1e-6:
            ankle_unit = np.zeros(3, dtype=np.float32)
        else:
            ankle_unit = ankle_vec / av_norm

        ankle_vector_x = float(ankle_unit[0])
        ankle_vector_z = float(ankle_unit[2])

        # d) average_hip_angle  (angle(shoulder, hip, knee) izq y der)
        right_hip_angle = calculate_angle(rs, rh, rk)
        left_hip_angle = calculate_angle(ls, lh, lk)
        average_hip_angle = float(np.nanmean(
            [right_hip_angle, left_hip_angle]))

        # e) average_knee_angle (angle(hip, knee, ankle) izq y der)
        right_knee_angle = calculate_angle(rh, rk, ra)
        left_knee_angle = calculate_angle(lh, lk, la)
        average_knee_angle = float(np.nanmean(
            [right_knee_angle, left_knee_angle]))

        feature_vector = np.array(
            [
                normalized_leg_length,
                shoulder_vector_x,
                shoulder_vector_z,
                ankle_vector_x,
                ankle_vector_z,
                average_hip_angle,
                average_knee_angle,
            ],
            dtype=np.float32,
        )

        feature_vector = np.nan_to_num(
            feature_vector, nan=0.0, posinf=0.0, neginf=0.0
        )

        if feature_vector.shape[0] != len(FEATURE_COLUMNS_V2):
            if DEBUG_FEATURES:
                print(
                    f"Error: Feature vector v2 con tamaño incorrecto. "
                    f"Esperado {len(FEATURE_COLUMNS_V2)}, obtenido {feature_vector.shape[0]}"
                )
            return None

        return feature_vector

    except KeyError as e:
        if DEBUG_FEATURES:
            print(
                f"Error en calculate_frame_features_v2: landmark faltante {e}")
        return None
    except Exception as e:
        if DEBUG_FEATURES:
            print(f"Error inesperado en calculate_frame_features_v2: {e}")
        return None
