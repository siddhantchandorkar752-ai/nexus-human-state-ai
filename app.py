import gradio as gr
import cv2
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vision_core import NexusVisionCore
from models.emotion_engine import NexusEmotionEngine
from models.risk_scorer import NexusRiskScorer

vision  = NexusVisionCore()
emotion = NexusEmotionEngine()
risk    = NexusRiskScorer()

def process_image(image):
    if image is None:
        return None, "No image provided"
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (1280, 720))
    display, vision_r  = vision.process_frame(frame)
    emotion_r = emotion.analyze(frame, vision_r)
    risk_r    = risk.compute(vision_r, emotion_r)
    H, W = display.shape[:2]
    cv2.rectangle(display,(0,0),(W,40),(10,10,10),-1)
    cv2.putText(display,
        f"NEXUS v2 | Eye:{vision_r['eye_state']} | Blinks:{vision_r['blink_count']} | Emotion:{emotion_r['dominant_emotion'].upper()} | Stress:{emotion_r['stress_level']:.2f} | Posture:{vision_r['posture_score']}/100",
        (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,180), 1)
    risk_color = risk_r["color"]
    cv2.rectangle(display,(0,H-60),(320,H),(15,15,15),-1)
    cv2.putText(display,f"RISK: {risk_r['level']} | Score: {risk_r['score']:.3f}",
        (8,H-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
    cv2.putText(display, risk_r["action"],
        (8,H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, risk_color, 1)
    result_text = f"""
NEXUS Analysis:
- Persons: {len(vision_r['persons'])}
- Eye State: {vision_r['eye_state']}
- Blink Count: {vision_r['blink_count']}
- Emotion: {emotion_r['dominant_emotion'].upper()}
- Stress Level: {emotion_r['stress_level']:.2f}
- Posture Score: {vision_r['posture_score']}/100
- Posture Issues: {', '.join(vision_r['posture_issues']) if vision_r['posture_issues'] else 'None'}
- Heart Rate: {vision_r['heart_rate']} BPM
- Risk Level: {risk_r['level']}
- Risk Score: {risk_r['score']:.3f}
- Action: {risk_r['action']}
    """
    output = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    return output, result_text

with gr.Blocks(title="NEXUS — Human State Intelligence AI", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style='text-align:center; padding:20px; background:linear-gradient(135deg,#0f0f0f,#1a1a2e); border-radius:12px; margin-bottom:20px'>
        <h1 style='color:#00ff9f; font-size:2.5em; margin:0'>🧠 NEXUS</h1>
        <p style='color:#888; font-size:1.1em'>Human State Intelligence AI — Real-time Emotion, Posture & Risk Analysis</p>
        <p style='color:#555'>Built by Siddhant Chandorkar</p>
    </div>
    """)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image / Webcam", sources=["webcam","upload"])
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="NEXUS Output")
            output_text  = gr.Textbox(label="Analysis Report", lines=15)
    analyze_btn.click(process_image, inputs=input_image, outputs=[output_image, output_text])
    gr.HTML("<div style='text-align:center;color:#444;padding:10px'>NEXUS v2 — Action Units | Posture Analysis | rPPG Heart Rate | Blink Detection</div>")

demo.launch(server_name="0.0.0.0", server_port=7860)
