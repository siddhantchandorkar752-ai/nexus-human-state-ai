import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from models.vision_core import NexusVisionCore
from models.emotion_engine import NexusEmotionEngine
from models.risk_scorer import NexusRiskScorer

class NEXUS:
    def __init__(self, source=0):
        print("\n" + "="*50)
        print("   NEXUS — Human State Intelligence AI v2")
        print("="*50)
        self.vision  = NexusVisionCore()
        self.emotion = NexusEmotionEngine()
        self.risk    = NexusRiskScorer()
        self.source  = source
        self._last_emotion = {
            "dominant_emotion": "neutral",
            "emotion_scores": {},
            "stress_level": 0.0,
            "valence": "neutral"
        }
        print("ALL SYSTEMS ONLINE\n")

    def draw_hud(self, frame, vision_r, emotion_r, risk_r):
        H, W = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame,(0,0),(W,40),(10,10,10),-1)
        cv2.putText(frame,
            f"NEXUS v2  |  Persons:{len(vision_r['persons'])}  |  Eye:{vision_r['eye_state']}  |  Blinks:{vision_r['blink_count']}  |  HR:{vision_r['heart_rate']} BPM  |  Emotion:{emotion_r['dominant_emotion'].upper()}  |  Stress:{emotion_r['stress_level']:.2f}",
            (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,180), 1)

        # Posture panel
        pscore  = vision_r.get("posture_score", 100)
        pissues = vision_r.get("posture_issues", [])
        pcolor  = (0,255,0) if pscore>70 else (0,165,255) if pscore>40 else (0,0,255)
        cv2.rectangle(frame,(0,45),(220,45+20+len(pissues)*18),(15,15,15),-1)
        cv2.putText(frame, f"POSTURE: {pscore}/100",
            (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pcolor, 1)
        for i,issue in enumerate(pissues):
            cv2.putText(frame, f"  ! {issue}",
                (5, 78+i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)

        # Heart rate panel
        hr = vision_r.get("heart_rate", 0)
        hr_color = (0,255,100) if 60<=hr<=100 else (0,165,255) if hr>0 else (100,100,100)
        cv2.rectangle(frame,(W-160,45),(W,80),(15,15,15),-1)
        cv2.putText(frame, f"HR: {hr} BPM",
            (W-155,68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hr_color, 2)

        # Emotion bars
        em_scores = emotion_r.get("emotion_scores",{})
        if em_scores:
            x_start = W-210
            for i,(em,sc) in enumerate(em_scores.items()):
                y = 90 + i*22
                bar = int(sc*180)
                cv2.rectangle(frame,(x_start,y),(x_start+bar,y+14),
                    self.emotion.get_emotion_color(em),-1)
                cv2.putText(frame,f"{em[:4]}:{sc:.2f}",
                    (x_start-70,y+12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)

        # Risk panel bottom
        risk_color = risk_r["color"]
        cv2.rectangle(frame,(0,H-85),(320,H),(15,15,15),-1)
        cv2.putText(frame, f"RISK: {risk_r['level']}",
            (8,H-58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_color, 2)
        cv2.putText(frame, f"Score: {risk_r['score']:.3f}",
            (8,H-32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 1)
        cv2.putText(frame, risk_r["action"],
            (8,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, risk_color, 1)

        # Risk bar
        bar_w = int(risk_r["score"] * (W-20))
        cv2.rectangle(frame,(10,H-95),(W-10,H-87),(50,50,50),-1)
        cv2.rectangle(frame,(10,H-95),(10+bar_w,H-87),risk_color,-1)

        # HIGH RISK flash
        if risk_r["level"] == "HIGH":
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(W,H),(0,0,180),-1)
            cv2.addWeighted(overlay,0.12,frame,0.88,0,frame)
            cv2.putText(frame,"!! HIGH RISK DETECTED !!",(W//2-220,H//2),
                cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)

        return frame

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open source {self.source}")
            return

        if isinstance(self.source, str):
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_W = 1280
            out_H = int(H * out_W / W)
            if out_H > 720: out_H=720; out_W=int(W*out_H/H)
        else:
            out_W, out_H = 1280, 720
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, out_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, out_H)

        cv2.namedWindow("NEXUS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("NEXUS", out_W, out_H)
        frame_count = 0
        print("Press Q to quit | S for screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.resize(frame,(out_W,out_H))

            display, vision_r = self.vision.process_frame(frame)

            if frame_count % 3 == 0:
                self._last_emotion = self.emotion.analyze(frame, vision_r)

            risk_r = self.risk.compute(vision_r, self._last_emotion)
            display = self.draw_hud(display, vision_r, self._last_emotion, risk_r)

            cv2.imshow("NEXUS", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"nexus_{frame_count}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

        cap.release()
        cv2.destroyAllWindows()
        print("NEXUS shutdown.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()
    source = 0 if args.source=="0" else args.source
    nexus = NEXUS(source=source)
    nexus.run()
