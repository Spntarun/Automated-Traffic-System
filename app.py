import os
import time
import threading
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# Path to the trained YOLO model and Hugging Face ViT model
MODEL_PATH = "models/emergency_model.pt"
VIT_ACCIDENT_MODEL_NAME = "tiya1012/vit-accident-image"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
try:
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        model.to("cuda")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

try:
    print(f"Loading ViT Accident model from {VIT_ACCIDENT_MODEL_NAME} on {DEVICE}...")
    accident_processor = AutoImageProcessor.from_pretrained(VIT_ACCIDENT_MODEL_NAME)
    accident_model = AutoModelForImageClassification.from_pretrained(VIT_ACCIDENT_MODEL_NAME).to(DEVICE)
    accident_model.eval()
    print("ViT Accident model loaded successfully on", DEVICE)
except Exception as e:
    print(f"Error loading ViT accident model: {e}")
    accident_processor = None
    accident_model = None

def detect_accident_in_frame(frame_bgr):
    """
    Evaluates a single BGR frame using the ViT accident classification model.
    tiya1012/vit-accident-image maps:
      index 0 (LABEL_0): No Accident
      index 1 (LABEL_1): Accident
    Returns: (is_accident: bool, accident_prob: float, pred_idx: int)
    """
    if accident_model is None or accident_processor is None:
        return False, 0.0, 0

    try:
        rgb_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = accident_processor(images=rgb_img, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = accident_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            accident_prob = probs[1].item()

        # Flag as accident if predicted class is 1 or accident probability is elevated (>= 0.35)
        is_accident = (pred_idx == 1 or accident_prob >= 0.35)
        return is_accident, accident_prob, pred_idx
    except Exception as e:
        print(f"Error in ViT accident detection: {e}")
        return False, 0.0, 0

EMERGENCY_CLASSES = {'ambulance', 'ambulances', 'fire_truck', 'police', 'polices', 'army', 'firesymbol'}

def process_single_frame(frame_bgr):
    """
    Runs ViT accident classification and YOLO vehicle/emergency detection on a frame.
    Returns:
      is_acc (bool), acc_prob (float),
      emergency_count (int), max_emergency_conf (float),
      vehicle_count (int), emergency_names (list),
      boxes_info (list of tuples: (x1, y1, x2, y2, class_name, is_emergency, conf))
    """
    is_acc, acc_prob, pred_idx = detect_accident_in_frame(frame_bgr)
    
    emergency_count = 0
    max_emergency_conf = 0.0
    vehicle_count = 0
    detected_emergency_names = []
    boxes_info = []

    if model:
        results = model.predict(source=frame_bgr, save=False, conf=0.18, imgsz=416, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id].lower()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check for emergency vehicle with sufficient confidence
                is_emergency = (class_name in EMERGENCY_CLASSES and conf >= 0.28)
                if is_emergency:
                    emergency_count += 1
                    detected_emergency_names.append(class_name)
                    if conf > max_emergency_conf:
                        max_emergency_conf = conf
                else:
                    vehicle_count += 1
                    
                boxes_info.append((x1, y1, x2, y2, class_name, is_emergency, conf))

    return is_acc, acc_prob, emergency_count, max_emergency_conf, vehicle_count, detected_emergency_names, boxes_info

def determine_event_type(e_names, max_e_conf, max_acc_prob, is_acc):
    """
    Disambiguates between genuine emergency vehicles and accident scenes:
    - Dedicated emergency vehicles (ambulances, fire trucks, police) take precedence over background ViT accident noise.
    - High-confidence accident crashes without emergency vehicles are flagged as ACCIDENT.
    """
    has_strong_emergency = any(name in ['ambulance', 'fire_truck'] for name in e_names) and max_e_conf >= 0.32
    has_police_emergency = ('police' in e_names or 'army' in e_names) and max_e_conf >= 0.40
    
    if has_strong_emergency or has_police_emergency:
        return 'EMERGENCY'
    elif max_acc_prob >= 0.50 and max_acc_prob > max_e_conf:
        return 'ACCIDENT'
    elif max_e_conf >= 0.30 and not is_acc:
        return 'EMERGENCY'
    elif is_acc and max_acc_prob >= 0.50:
        return 'ACCIDENT'
    else:
        return 'NORMAL'

# Global state for traffic management
lanes = ['N', 'S', 'E', 'W']

traffic_state = {
    'lanes': {
        'N': {'vehicle_count': 0, 'emergency_count': 0, 'accident_count': 0, 'signal': 'Red', 'pedestrian': 'Red'},
        'S': {'vehicle_count': 0, 'emergency_count': 0, 'accident_count': 0, 'signal': 'Red', 'pedestrian': 'Red'},
        'E': {'vehicle_count': 0, 'emergency_count': 0, 'accident_count': 0, 'signal': 'Red', 'pedestrian': 'Red'},
        'W': {'vehicle_count': 0, 'emergency_count': 0, 'accident_count': 0, 'signal': 'Red', 'pedestrian': 'Red'}
    },
    'active_lane': None,
    'override_active': False,
    'pedestrian_override': None,
    'phase_timer': 0,
    'current_phase': 'Red'
}

state_lock = threading.Lock()

def traffic_controller_loop():
    global traffic_state
    current_lane_idx = -1
    
    while True:
        with state_lock:
            # 1. Handle overrides
            if traffic_state['pedestrian_override']:
                ped_lane = traffic_state['pedestrian_override']
                for l in lanes:
                    traffic_state['lanes'][l]['signal'] = 'Red'
                    traffic_state['lanes'][l]['pedestrian'] = 'Red'
                
                traffic_state['lanes'][ped_lane]['pedestrian'] = 'Green'
                traffic_state['active_lane'] = ped_lane
                traffic_state['phase_timer'] = 5
                traffic_state['current_phase'] = 'PedestrianOverride'
                traffic_state['pedestrian_override'] = None
                
            else:
                # 2. Check for Emergency Corridor (Priority 1) or Accident Response (Priority 2)
                priority_lane = None
                phase_type = None

                # Check Emergency Vehicles first
                for lane in lanes:
                    if traffic_state['lanes'][lane]['emergency_count'] > 0:
                        priority_lane = lane
                        phase_type = 'Emergency'
                        break

                # Check Accidents second if no emergency vehicle
                if not priority_lane:
                    for lane in lanes:
                        if traffic_state['lanes'][lane]['accident_count'] > 0:
                            priority_lane = lane
                            phase_type = 'AccidentResponse'
                            break
                
                if priority_lane and traffic_state['current_phase'] not in ['Emergency', 'AccidentResponse']:
                    for l in lanes:
                        traffic_state['lanes'][l]['signal'] = 'Red'
                        traffic_state['lanes'][l]['pedestrian'] = 'Red'
                        
                    traffic_state['active_lane'] = priority_lane
                    traffic_state['lanes'][priority_lane]['signal'] = 'Green'
                    traffic_state['phase_timer'] = 30
                    traffic_state['current_phase'] = phase_type
                    
                else:
                    # 3. Normal Auto-Optimize Loop (Tick-based)
                    if traffic_state['phase_timer'] <= 0:
                        if traffic_state['current_phase'] in ['Green', 'Emergency', 'AccidentResponse', 'PedestrianOverride', 'Override']:
                            # Transition to Yellow
                            active = traffic_state['active_lane']
                            if active:
                                traffic_state['lanes'][active]['signal'] = 'Yellow'
                                traffic_state['lanes'][active]['pedestrian'] = 'Red'
                                # Clear emergency/accident counts when the priority clearing period ends
                                traffic_state['lanes'][active]['emergency_count'] = 0
                                traffic_state['lanes'][active]['accident_count'] = 0
                            traffic_state['phase_timer'] = 2
                            traffic_state['current_phase'] = 'Yellow'
                            
                        else:
                            # Transition from Yellow/Red to Next Green
                            active = traffic_state['active_lane']
                            if active:
                                traffic_state['lanes'][active]['signal'] = 'Red'
                            
                            current_lane_idx = (current_lane_idx + 1) % len(lanes)
                            next_lane = lanes[current_lane_idx]
                            
                            traffic_state['active_lane'] = next_lane
                            traffic_state['lanes'][next_lane]['signal'] = 'Green'
                            traffic_state['lanes'][next_lane]['pedestrian'] = 'Green'
                            
                            count = traffic_state['lanes'][next_lane]['vehicle_count']
                            # Minimum 30 seconds, dynamically adding 2 seconds per vehicle, max 90 seconds
                            green_time = min(90, max(30, 30 + (count * 2)))
                            traffic_state['phase_timer'] = green_time
                            traffic_state['current_phase'] = 'Green'
                            
        # Sleep 1 second for the tick
        time.sleep(1)
        
        # Decrement timer
        with state_lock:
            if traffic_state['phase_timer'] > 0:
                traffic_state['phase_timer'] -= 1

# Start background thread
controller_thread = threading.Thread(target=traffic_controller_loop, daemon=True)
controller_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    with state_lock:
        return jsonify(traffic_state)

@app.route('/api/upload/<lane>', methods=['POST'])
def upload_media(lane):
    if lane not in lanes:
        return jsonify({'error': 'Invalid lane'}), 400
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and model:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{lane}_{filename}")
        file.save(filepath)
        
        # Check if it is a video or image
        ext = filename.rsplit('.', 1)[-1].lower()
        selected_data = None

        if ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            cap = cv2.VideoCapture(filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Dense sampling across video (up to 50 candidate frames)
            if total_frames <= 50:
                sample_step = 1
            else:
                sample_step = max(2, total_frames // 50)
            
            frame_idx = 0
            first_frame_data = None
            middle_frame_data = None
            middle_target_idx = total_frames // 2 if total_frames > 0 else 0
            
            best_accident_data = None
            max_accident_prob = 0.0
            
            best_emergency_data = None
            max_emergency_score = 0.0

            while cap.isOpened():
                ret, current_frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_step == 0 or frame_idx == 0:
                    is_acc, acc_prob, e_count, max_e_conf, v_count, e_names, boxes = process_single_frame(current_frame)
                    
                    frame_data = {
                        'frame': current_frame.copy(),
                        'frame_idx': frame_idx,
                        'is_acc': is_acc,
                        'acc_prob': acc_prob,
                        'e_count': e_count,
                        'max_e_conf': max_e_conf,
                        'v_count': v_count,
                        'e_names': e_names,
                        'boxes': boxes
                    }
                    
                    if first_frame_data is None:
                        first_frame_data = frame_data
                        
                    if middle_frame_data is None or abs(frame_idx - middle_target_idx) < abs(middle_frame_data['frame_idx'] - middle_target_idx):
                        middle_frame_data = frame_data
                        
                    # 1. Track the exact frame with the HIGHEST emergency vehicle confidence
                    if e_count > 0 and (max_e_conf > max_emergency_score or best_emergency_data is None):
                        max_emergency_score = max_e_conf
                        best_emergency_data = frame_data

                    # 2. Track the exact frame with the HIGHEST accident probability
                    if is_acc and acc_prob > max_accident_prob:
                        max_accident_prob = acc_prob
                        best_accident_data = frame_data

                frame_idx += 1

            cap.release()

            # Determine global video event type
            video_e_names = best_emergency_data['e_names'] if best_emergency_data else []
            event_type = determine_event_type(
                video_e_names,
                max_emergency_score,
                max_accident_prob,
                best_accident_data['is_acc'] if best_accident_data else False
            )

            # EXACT KEYFRAME SELECTION:
            if event_type == 'EMERGENCY' and best_emergency_data is not None:
                selected_data = best_emergency_data
                is_emergency_event = True
                is_accident_event = False
            elif event_type == 'ACCIDENT' and best_accident_data is not None:
                selected_data = best_accident_data
                is_emergency_event = False
                is_accident_event = True
            elif middle_frame_data is not None:
                selected_data = middle_frame_data
                is_emergency_event = False
                is_accident_event = False
            elif first_frame_data is not None:
                selected_data = first_frame_data
                is_emergency_event = False
                is_accident_event = False
            else:
                return jsonify({'error': 'Failed to read video'}), 400
        else:
            frame = cv2.imread(filepath)
            if frame is None:
                return jsonify({'error': 'Failed to read image'}), 400
            
            is_acc, acc_prob, e_count, max_e_conf, v_count, e_names, boxes = process_single_frame(frame)
            selected_data = {
                'frame': frame,
                'frame_idx': 0,
                'is_acc': is_acc,
                'acc_prob': acc_prob,
                'e_count': e_count,
                'max_e_conf': max_e_conf,
                'v_count': v_count,
                'e_names': e_names,
                'boxes': boxes
            }
            
            event_type = determine_event_type(e_names, max_e_conf, acc_prob, is_acc)
            is_emergency_event = (event_type == 'EMERGENCY')
            is_accident_event = (event_type == 'ACCIDENT')

        # Annotate display frame
        display_frame = selected_data['frame'].copy()
        h, w = display_frame.shape[:2]

        # 1. Draw Bounding Boxes for detected vehicles and emergency vehicles
        for (x1, y1, x2, y2, class_name, is_emergency, conf) in selected_data['boxes']:
            if is_emergency:
                # Orange/Red prominent box for emergency vehicle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
                label_text = f"EMERGENCY: {class_name.upper()} ({int(conf * 100)}%)"
                cv2.rectangle(display_frame, (x1, max(0, y1 - 25)), (x1 + len(label_text) * 11, y1), (0, 140, 255), -1)
                cv2.putText(display_frame, label_text, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            else:
                # Green box for normal vehicle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{class_name}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # 2. Draw Top Banners & Alerts on the Exact Keyframe
        if is_emergency_event:
            banner_height = max(40, int(h * 0.12))
            cv2.rectangle(display_frame, (0, 0), (w, banner_height), (0, 110, 245), -1)
            e_str = ", ".join(set(selected_data['e_names'])).upper()
            banner_text = f"PRIORITY: EMERGENCY VEHICLE ({e_str}) [Frame #{selected_data['frame_idx']}]"
            font_scale = max(0.48, min(0.85, w / 700))
            cv2.putText(display_frame, banner_text, (15, int(banner_height * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 140, 255), 5)
        elif is_accident_event:
            banner_height = max(40, int(h * 0.12))
            cv2.rectangle(display_frame, (0, 0), (w, banner_height), (0, 0, 210), -1)
            banner_text = f"CRITICAL: ACCIDENT DETECTED (ViT {int(selected_data['acc_prob'] * 100)}%) [Frame #{selected_data['frame_idx']}]"
            font_scale = max(0.48, min(0.85, w / 700))
            cv2.putText(display_frame, banner_text, (15, int(banner_height * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(display_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 5)

        # Save annotated keyframe image for frontend
        output_path = os.path.join('static', 'outputs', f'{lane}.jpg')
        cv2.imwrite(output_path, display_frame)
                    
        vehicle_count = selected_data['v_count']
        emergency_count = selected_data['e_count'] if is_emergency_event else 0
        accident_count = 1 if is_accident_event else 0

        with state_lock:
            traffic_state['lanes'][lane]['vehicle_count'] = vehicle_count
            traffic_state['lanes'][lane]['emergency_count'] = emergency_count
            traffic_state['lanes'][lane]['accident_count'] = accident_count
            
        return jsonify({
            'lane': lane,
            'vehicle_count': vehicle_count,
            'emergency_count': emergency_count,
            'accident_count': accident_count,
            'accident_confidence': selected_data['acc_prob'] if is_accident_event else 0.0,
            'emergency_detected': is_emergency_event,
            'accident_detected': is_accident_event,
            'frame_index': selected_data['frame_idx']
        })
        
    return jsonify({'error': 'Model not loaded or internal error'}), 500

@app.route('/api/override', methods=['POST'])
def override_signal():
    data = request.json
    lane = data.get('lane')
    active = data.get('active', False)
    
    with state_lock:
        if not active:
            # Resume auto instantly
            traffic_state['override_active'] = False
            traffic_state['phase_timer'] = 0 
        elif lane in lanes:
            traffic_state['override_active'] = False # Don't freeze the loop
            for l in lanes:
                traffic_state['lanes'][l]['signal'] = 'Red'
                traffic_state['lanes'][l]['pedestrian'] = 'Red'
            traffic_state['lanes'][lane]['signal'] = 'Green'
            traffic_state['lanes'][lane]['pedestrian'] = 'Green'
            traffic_state['active_lane'] = lane
            traffic_state['phase_timer'] = 30
            traffic_state['current_phase'] = 'Override'
            
    return jsonify({'success': True, 'state': traffic_state})

@app.route('/api/override_pedestrian', methods=['POST'])
def override_pedestrian():
    data = request.json
    lane = data.get('lane')
    
    with state_lock:
        traffic_state['pedestrian_override'] = lane
            
    return jsonify({'success': True, 'state': traffic_state})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
