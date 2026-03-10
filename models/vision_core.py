import cv2
import numpy as np
from ultralytics import YOLO

class NexusVisionCore:
    def __init__(self):
        print("Loading NEXUS Vision Core...")
        self.yolo_det  = YOLO("yolo11n.pt")
        self.yolo_pose = YOLO("yolo11n-pose.pt")
        
        # Eye blink tracking
        self.blink_count    = 0
        self.blink_state    = "OPEN"
        self.closed_frames  = 0
        self.ear_history    = []
        
        # rPPG
        self.rppg_buffer    = []
        self.rppg_fps       = 30
        self.heart_rate     = 0
        
        # Face geometry
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("Vision Core Ready!")

    def get_ear(self, eye_pts):
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.3

    def compute_rppg(self, frame, face_box):
        x1,y1,x2,y2 = face_box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return self.heart_rate
        green = np.mean(roi[:,:,1])
        self.rppg_buffer.append(green)
        if len(self.rppg_buffer) > 150:
            self.rppg_buffer.pop(0)
        if len(self.rppg_buffer) >= 60:
            signal = np.array(self.rppg_buffer)
            signal = signal - np.mean(signal)
            fft    = np.abs(np.fft.rfft(signal))
            freqs  = np.fft.rfftfreq(len(signal), 1.0/self.rppg_fps)
            mask   = (freqs >= 0.8) & (freqs <= 3.0)
            if mask.any():
                peak = freqs[mask][np.argmax(fft[mask])]
                self.heart_rate = int(peak * 60)
        return self.heart_rate

    def compute_posture(self, keypoints):
        score  = 100
        issues = []
        try:
            nose    = keypoints[0][:2]
            l_shldr = keypoints[5][:2]
            r_shldr = keypoints[6][:2]
            l_hip   = keypoints[11][:2]
            r_hip   = keypoints[12][:2]
            l_ear   = keypoints[3][:2]
            r_ear   = keypoints[4][:2]

            # Shoulder level
            shldr_tilt = abs(l_shldr[1] - r_shldr[1])
            if shldr_tilt > 20:
                score -= 20
                issues.append("Shoulder tilt")

            # Head forward
            mid_shldr = (l_shldr + r_shldr) / 2
            mid_ear   = (l_ear + r_ear) / 2
            fwd_lean  = mid_ear[0] - mid_shldr[0]
            if abs(fwd_lean) > 30:
                score -= 25
                issues.append("Head forward")

            # Spine — shoulder to hip alignment
            mid_hip  = (l_hip + r_hip) / 2
            spine_dx = abs(mid_shldr[0] - mid_hip[0])
            if spine_dx > 40:
                score -= 20
                issues.append("Spine lean")

            # Head down
            head_down = nose[1] > mid_shldr[1] - 30
            if head_down:
                score -= 15
                issues.append("Head down")

        except:
            score = 50
        return max(score, 0), issues

    def get_face_emotions(self, frame, face_box):
        emotions = {
            "happy":    0.0,
            "sad":      0.0,
            "angry":    0.0,
            "surprised":0.0,
            "fearful":  0.0,
            "disgusted":0.0,
            "neutral":  0.5
        }
        try:
            x1,y1,x2,y2 = face_box
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return emotions, "neutral"
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Eye region brightness
            eye_region = gray[int(h*0.2):int(h*0.5), :]
            eye_bright  = np.mean(eye_region)

            # Mouth region
            mouth_region = gray[int(h*0.65):int(h*0.9), int(w*0.2):int(w*0.8)]
            mouth_bright  = np.mean(mouth_region)

            # Forehead
            fore_region = gray[0:int(h*0.2), :]
            fore_bright  = np.mean(fore_region)

            # Brightness variance — stress indicator
            variance = np.std(gray)

            # Simple geometry rules
            if mouth_bright > eye_bright + 10:
                emotions["happy"]   = 0.7
                emotions["neutral"] = 0.2
            elif eye_bright < 80:
                emotions["sad"]     = 0.5
                emotions["fearful"] = 0.3
                emotions["neutral"] = 0.2
            elif variance > 60:
                emotions["angry"]    = 0.5
                emotions["disgusted"]= 0.3
                emotions["neutral"]  = 0.2
            elif fore_bright > 150:
                emotions["surprised"] = 0.6
                emotions["neutral"]   = 0.3
            else:
                emotions["neutral"] = 0.8

            dominant = max(emotions, key=emotions.get)
        except:
            dominant = "neutral"
        return emotions, dominant

    def process_frame(self, frame):
        H, W = frame.shape[:2]
        output = frame.copy()
        results = {
            "persons":      [],
            "face_box":     None,
            "eye_state":    "OPEN",
            "ear_value":    0.3,
            "blink_count":  self.blink_count,
            "heart_rate":   self.heart_rate,
            "posture_score":100,
            "posture_issues":[],
            "face_emotions":{},
            "dominant_face_emotion":"neutral",
            "keypoints":    None
        }

        # Detection
        det = self.yolo_det(frame, verbose=False, classes=[0])[0]
        if det.boxes is not None:
            for box in det.boxes:
                if float(box.conf[0]) > 0.4:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    results["persons"].append([x1,y1,x2,y2])
                    cv2.rectangle(output,(x1,y1),(x2,y2),(0,255,100),2)
                    cv2.putText(output,f"Person {float(box.conf[0]):.2f}",
                        (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,100),1)

        # Pose
        pose = self.yolo_pose(frame, verbose=False)[0]
        if pose.keypoints is not None and len(pose.keypoints.data) > 0:
            kpts = pose.keypoints.data[0].cpu().numpy()
            results["keypoints"] = kpts

            # Posture
            pscore, pissues = self.compute_posture(kpts)
            results["posture_score"]  = pscore
            results["posture_issues"] = pissues

            # EAR from pose eye keypoints
            l_eye = kpts[1][:2]
            r_eye = kpts[2][:2]
            eye_dist = np.linalg.norm(l_eye - r_eye)
            nose     = kpts[0][:2]
            eye_mid  = (l_eye + r_eye) / 2
            eye_nose = np.linalg.norm(eye_mid - nose)
            ear_approx = eye_nose / (eye_dist + 1e-5)
            results["ear_value"] = round(ear_approx, 3)

            # Blink detection
            if ear_approx < 0.6:
                self.closed_frames += 1
                results["eye_state"] = "CLOSED"
                if self.closed_frames == 2:
                    self.blink_count += 1
            else:
                self.closed_frames = 0
                results["eye_state"] = "OPEN"
            results["blink_count"] = self.blink_count

            # Draw skeleton
            CONNECTIONS = [
                (5,6),(5,7),(7,9),(6,8),(8,10),
                (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
            ]
            for a,b in CONNECTIONS:
                if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                    pt1 = tuple(map(int, kpts[a][:2]))
                    pt2 = tuple(map(int, kpts[b][:2]))
                    cv2.line(output, pt1, pt2, (0,255,180), 2)
            for i,kp in enumerate(kpts):
                if kp[2] > 0.3:
                    cv2.circle(output, tuple(map(int,kp[:2])), 4, (255,100,0), -1)

        # Face detection + rPPG + face emotions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        if len(faces) > 0:
            x,y,w,h = faces[0]
            face_box = (x, y, x+w, y+h)
            results["face_box"] = face_box
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,200,255),1)

            # rPPG
            results["heart_rate"] = self.compute_rppg(frame, face_box)

            # Face emotions
            em, dom = self.get_face_emotions(frame, face_box)
            results["face_emotions"]         = em
            results["dominant_face_emotion"] = dom

        return output, results
