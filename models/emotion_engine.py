import cv2
import numpy as np
from scipy.signal import butter, filtfilt

class NexusEmotionEngine:
    def __init__(self):
        print("Loading Emotion Engine...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        self.history       = []
        self.au_history    = []
        self.rppg_buffer   = []
        self.prev_gray     = None
        self.optical_flow_pts = None
        print("Emotion Engine Ready!")

    def detect_action_units(self, face_gray, face_color):
        h, w = face_gray.shape
        aus = {}

        # Regions
        forehead   = face_gray[0:int(h*0.25), int(w*0.2):int(w*0.8)]
        l_eye_r    = face_gray[int(h*0.2):int(h*0.45), int(w*0.05):int(w*0.45)]
        r_eye_r    = face_gray[int(h*0.2):int(h*0.45), int(w*0.55):int(w*0.95)]
        nose_r     = face_gray[int(h*0.4):int(h*0.65), int(w*0.3):int(w*0.7)]
        mouth_r    = face_gray[int(h*0.65):int(h*0.92), int(w*0.15):int(w*0.85)]
        l_cheek    = face_gray[int(h*0.45):int(h*0.7), int(w*0.05):int(w*0.35)]
        r_cheek    = face_gray[int(h*0.45):int(h*0.7), int(w*0.65):int(w*0.95)]

        # AU1+2 — inner/outer brow raise (forehead wrinkles)
        fore_var = np.std(forehead) if forehead.size > 0 else 0
        aus["AU1_brow_raise"] = min(fore_var / 30.0, 1.0)

        # AU4 — brow furrow (dark vertical between brows)
        brow_center = face_gray[int(h*0.2):int(h*0.38), int(w*0.35):int(w*0.65)]
        aus["AU4_brow_furrow"] = 1.0 - min(np.mean(brow_center)/255.0 + 0.3, 1.0) if brow_center.size > 0 else 0

        # AU6 — cheek raise (brightness)
        cheek_bright = (np.mean(l_cheek) + np.mean(r_cheek)) / 2 if l_cheek.size > 0 else 128
        aus["AU6_cheek_raise"] = min(cheek_bright / 200.0, 1.0)

        # AU12 — lip corner pull (smile) — width of bright mouth region
        if mouth_r.size > 0:
            _, mouth_bin = cv2.threshold(mouth_r, 100, 255, cv2.THRESH_BINARY)
            mouth_width  = np.sum(mouth_bin[mouth_bin.shape[0]//2,:] > 0)
            aus["AU12_smile"] = min(mouth_width / (mouth_r.shape[1] * 0.8), 1.0)
        else:
            aus["AU12_smile"] = 0.0

        # AU15 — lip corner depress (sad)
        if mouth_r.size > 0:
            top_mouth = np.mean(mouth_r[:mouth_r.shape[0]//2,:])
            bot_mouth = np.mean(mouth_r[mouth_r.shape[0]//2:,:])
            aus["AU15_lip_depress"] = min(max(bot_mouth - top_mouth, 0) / 50.0, 1.0)
        else:
            aus["AU15_lip_depress"] = 0.0

        # AU17 — chin raise
        chin = face_gray[int(h*0.88):h, int(w*0.3):int(w*0.7)]
        aus["AU17_chin_raise"] = min(np.std(chin)/20.0, 1.0) if chin.size > 0 else 0

        # AU20 — lip stretch (fear)
        if mouth_r.size > 0:
            edges = cv2.Canny(mouth_r, 50, 150)
            aus["AU20_lip_stretch"] = min(np.sum(edges > 0) / (mouth_r.size * 0.15), 1.0)
        else:
            aus["AU20_lip_stretch"] = 0.0

        # AU23 — lip tighten (anger)
        if mouth_r.size > 0:
            mouth_dark = np.sum(mouth_r < 80) / mouth_r.size
            aus["AU23_lip_tighten"] = min(mouth_dark * 3, 1.0)
        else:
            aus["AU23_lip_tighten"] = 0.0

        # AU43 — eye closure
        l_mean = np.mean(l_eye_r) if l_eye_r.size > 0 else 128
        r_mean = np.mean(r_eye_r) if r_eye_r.size > 0 else 128
        eye_dark = 1.0 - min((l_mean + r_mean) / 2 / 128.0, 1.0)
        aus["AU43_eye_close"] = eye_dark

        return aus

    def aus_to_emotions(self, aus):
        emotions = {
            "happy":     0.0,
            "sad":       0.0,
            "angry":     0.0,
            "fearful":   0.0,
            "surprised": 0.0,
            "disgusted": 0.0,
            "neutral":   0.0
        }

        # FACS rules
        emotions["happy"]     = (aus["AU6_cheek_raise"] * 0.4 +
                                  aus["AU12_smile"] * 0.6)

        emotions["sad"]       = (aus["AU1_brow_raise"] * 0.3 +
                                  aus["AU15_lip_depress"] * 0.4 +
                                  aus["AU17_chin_raise"] * 0.3)

        emotions["angry"]     = (aus["AU4_brow_furrow"] * 0.4 +
                                  aus["AU23_lip_tighten"] * 0.4 +
                                  aus["AU17_chin_raise"] * 0.2)

        emotions["fearful"]   = (aus["AU1_brow_raise"] * 0.3 +
                                  aus["AU20_lip_stretch"] * 0.4 +
                                  aus["AU43_eye_close"] * 0.3)

        emotions["surprised"] = (aus["AU1_brow_raise"] * 0.4 +
                                  aus["AU20_lip_stretch"] * 0.3 +
                                  (1.0 - aus["AU43_eye_close"]) * 0.3)

        emotions["disgusted"] = (aus["AU4_brow_furrow"] * 0.3 +
                                  aus["AU23_lip_tighten"] * 0.3 +
                                  aus["AU15_lip_depress"] * 0.4)

        # Normalize
        total = sum(emotions.values())
        if total < 0.3:
            emotions["neutral"] = 1.0
        else:
            for k in emotions:
                emotions[k] = round(emotions[k] / (total + 1e-5), 3)

        dominant = max(emotions, key=emotions.get)
        return emotions, dominant

    def compute_optical_flow_stress(self, gray):
        stress = 0.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            stress = min(float(np.mean(mag)) / 2.0, 1.0)
        self.prev_gray = gray.copy()
        return stress

    def analyze(self, frame, vision_results):
        result = {
            "dominant_emotion": "neutral",
            "emotion_scores":   {},
            "stress_level":     0.0,
            "valence":          "neutral",
            "action_units":     {},
            "micro_movement":   0.0
        }

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 240))

        # Optical flow stress
        motion_stress = self.compute_optical_flow_stress(small)
        result["micro_movement"] = round(motion_stress, 3)

        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        if len(faces) > 0:
            x,y,w,h = faces[0]
            face_gray  = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            # Action Units
            aus = self.detect_action_units(face_gray, face_color)
            result["action_units"] = aus

            # Emotions from AUs
            emotions, dominant = self.aus_to_emotions(aus)
            result["dominant_emotion"] = dominant
            result["emotion_scores"]   = emotions

            # Stress
            posture_stress = max(0, (100 - vision_results.get("posture_score",100)) / 100)
            face_stress    = (emotions.get("angry",0) +
                              emotions.get("fearful",0) +
                              emotions.get("sad",0)) * 0.5
            total_stress   = min(posture_stress*0.35 + face_stress*0.35 + motion_stress*0.3, 1.0)
            result["stress_level"] = round(total_stress, 3)

            # Valence
            pos = emotions.get("happy",0) + emotions.get("surprised",0)*0.3
            neg = emotions.get("sad",0) + emotions.get("angry",0) + emotions.get("fearful",0)
            result["valence"] = "positive" if pos>neg else "negative" if neg>0.25 else "neutral"

        self.history.append(result["dominant_emotion"])
        if len(self.history) > 60:
            self.history.pop(0)

        return result

    def get_emotion_color(self, emotion):
        colors = {
            "happy":     (0,255,100),
            "neutral":   (200,200,200),
            "sad":       (255,100,0),
            "angry":     (0,0,255),
            "fearful":   (0,100,255),
            "surprised": (0,255,255),
            "disgusted": (100,0,255)
        }
        return colors.get(emotion, (200,200,200))
