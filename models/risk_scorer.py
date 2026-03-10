import numpy as np

class NexusRiskScorer:
    def __init__(self):
        print("Loading Risk Scorer...")
        self.W = {
            "eye_closed":   0.35,
            "stress":       0.25,
            "negative_val": 0.15,
            "no_face":      0.25
        }
        self.risk_history = []
        print("Risk Scorer Ready!")

    def compute(self, vision_results, emotion_results):
        score = 0.0
        if vision_results.get("eye_state") == "CLOSED":
            score += self.W["eye_closed"]
        stress = emotion_results.get("stress_level", 0)
        score += self.W["stress"] * stress
        if emotion_results.get("valence") == "negative":
            score += self.W["negative_val"]
        if vision_results.get("face_landmarks") is None:
            score += self.W["no_face"]
        score = min(score, 1.0)
        self.risk_history.append(score)
        if len(self.risk_history) > 30:
            self.risk_history.pop(0)
        avg_risk = np.mean(self.risk_history)
        if avg_risk < 0.3:
            level = "LOW"
            color = (0,255,0)
            action = "All systems normal"
        elif avg_risk < 0.6:
            level = "MEDIUM"
            color = (0,165,255)
            action = "Monitor closely"
        else:
            level = "HIGH"
            color = (0,0,255)
            action = "IMMEDIATE INTERVENTION REQUIRED"
        return {
            "score":   round(avg_risk, 3),
            "level":   level,
            "color":   color,
            "action":  action,
            "instant": round(score, 3)
        }
