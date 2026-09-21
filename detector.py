import os
import io
import base64
import cv2
import numpy as np
import tensorflow as tf

# Suppress noisy TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FF_DATASET_DIR = os.path.join(BASE_DIR, "FF_Dataset")

class DeepfakeDetector:
    def __init__(self):
        self.efficientnet_path = os.path.join(FF_DATASET_DIR, "efficientnet_model.h5")
        self.xception_path = os.path.join(FF_DATASET_DIR, "xception_model.h5")
        
        self.eff_model = None
        self.xception_model = None
        self.grad_model = None
        
        # Load OpenCV Face Detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.efficientnet_path):
            print(f"Loading EfficientNet model from: {self.efficientnet_path}")
            try:
                self.eff_model = tf.keras.models.load_model(self.efficientnet_path, compile=False)
                print("EfficientNet model loaded successfully!")
                
                # Build Grad-CAM model based on spatial attention multiply layer
                try:
                    target_layer_name = 'multiply_3'
                    self.grad_model = tf.keras.models.Model(
                        inputs=self.eff_model.inputs,
                        outputs=[self.eff_model.get_layer(target_layer_name).output, self.eff_model.output]
                    )
                    print("Grad-CAM model prepared successfully.")
                except Exception as ge:
                    print(f"Notice: Grad-CAM setup fell back to model output: {ge}")
                    self.grad_model = None
            except Exception as e:
                print(f"Error loading EfficientNet model: {e}")
        else:
            print(f"EfficientNet model not found at {self.efficientnet_path}")

        if os.path.exists(self.xception_path):
            print(f"Loading Xception model from: {self.xception_path}")
            try:
                self.xception_model = tf.keras.models.load_model(self.xception_path, compile=False)
                print("Xception model loaded successfully!")
            except Exception as e:
                print(f"Error loading Xception model: {e}")

    @property
    def is_ready(self):
        return self.eff_model is not None or self.xception_model is not None

    def apply_face_mask(self, image):
        """Apply elliptical mask to remove background artifacts, keeping facial features (Cell 30)."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.42), int(h * 0.47))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked

    def _generate_gradcam_heatmap(self, face_input_norm):
        """
        Compute Grad-CAM heatmap for a normalized face tensor of shape (1, 224, 224, 3).
        Returns a normalized 2D heatmap in [0, 1].
        """
        if self.grad_model is None:
            return None
        
        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(face_input_norm)
                loss = predictions[:, 0]
            
            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                return None
                
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return None

    def _overlay_heatmap_on_face(self, face_rgb, heatmap):
        """Overlay colorized Grad-CAM heatmap on RGB face crop."""
        h, w = face_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_u8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
        # Colorize using JET colormap
        heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(face_rgb, 0.6, heatmap_rgb, 0.4, 0)
        return overlay

    def _image_to_base64_data_url(self, image_rgb):
        """Encode an RGB numpy image to base64 JPEG data URL."""
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    def extract_faces(self, frame_bgr):
        """Detect faces in a BGR frame using OpenCV cascade classifier."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )
        return boxes

    def process_face_crop(self, face_rgb, apply_mask=False, generate_heatmap=True):
        """
        Runs model inference and optional Grad-CAM on a cropped RGB face.
        Returns:
            prob (float): probability of FAKE (0..1)
            face_data_url (str): base64 preview of face
            heatmap_data_url (str): base64 preview of Grad-CAM overlay
        """
        display_face = face_rgb.copy()
        if apply_mask:
            display_face = self.apply_face_mask(display_face)
            
        face_resized = cv2.resize(display_face, (224, 224))
        # EfficientNet model expects [0, 1] range as input layer 1 Rescaling(255.0) multiplies it back to [0, 255]
        face_norm = np.expand_dims(face_resized.astype(np.float32) / 255.0, axis=0)
        
        prob = 0.0
        if self.eff_model is not None and self.xception_model is not None:
            prob_eff = float(self.eff_model.predict(face_norm, verbose=0)[0][0])
            prob_xcep = float(self.xception_model.predict(face_norm, verbose=0)[0][0])
            # Equal ensemble 0.5 / 0.5 (optimal found in FaceForensics++ study)
            prob = 0.5 * prob_eff + 0.5 * prob_xcep
        elif self.eff_model is not None:
            prob = float(self.eff_model.predict(face_norm, verbose=0)[0][0])
        elif self.xception_model is not None:
            prob = float(self.xception_model.predict(face_norm, verbose=0)[0][0])

        face_data_url = self._image_to_base64_data_url(display_face)
        heatmap_data_url = None
        
        if generate_heatmap:
            heatmap = self._generate_gradcam_heatmap(face_norm)
            if heatmap is not None:
                overlay = self._overlay_heatmap_on_face(display_face, heatmap)
                heatmap_data_url = self._image_to_base64_data_url(overlay)

        return prob, face_data_url, heatmap_data_url

    def predict_image(self, image_data, apply_mask=False, generate_heatmap=True, threshold=0.5):
        """
        Analyze an image (path or raw bytes).
        Returns a dict with overall verdict, score, detected faces, and visualizations.
        """
        if isinstance(image_data, (bytes, bytearray)):
            nparr = np.frombuffer(image_data, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, str) and os.path.exists(image_data):
            frame_bgr = cv2.imread(image_data)
        else:
            raise ValueError("Invalid image input provided")

        if frame_bgr is None:
            raise ValueError("Could not decode image")

        h, w = frame_bgr.shape[:2]
        boxes = self.extract_faces(frame_bgr)
        
        faces_results = []
        
        # If no face is detected with cascade, evaluate center crop as fallback
        if len(boxes) == 0:
            min_dim = min(h, w)
            cy, cx = h // 2, w // 2
            half = min_dim // 2
            face_bgr = frame_bgr[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            prob, face_url, heatmap_url = self.process_face_crop(face_rgb, apply_mask, generate_heatmap)
            faces_results.append({
                "face_index": 1,
                "box": [0, 0, w, h],
                "fake_prob": round(prob, 4),
                "fake_prob_percent": round(prob * 100, 2),
                "face_url": face_url,
                "heatmap_url": heatmap_url,
                "fallback_crop": True
            })
        else:
            for idx, (x, y, fw, fh) in enumerate(boxes):
                # Add slight margin to capture face boundary artifacts
                margin = int(0.15 * max(fw, fh))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + fw + margin)
                y2 = min(h, y + fh + margin)
                
                face_bgr = frame_bgr[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                prob, face_url, heatmap_url = self.process_face_crop(face_rgb, apply_mask, generate_heatmap)
                faces_results.append({
                    "face_index": idx + 1,
                    "box": [int(x), int(y), int(fw), int(fh)],
                    "fake_prob": round(prob, 4),
                    "fake_prob_percent": round(prob * 100, 2),
                    "face_url": face_url,
                    "heatmap_url": heatmap_url,
                    "fallback_crop": False
                })

        avg_prob = float(np.mean([f["fake_prob"] for f in faces_results]))
        verdict = "FAKE" if avg_prob >= threshold else "REAL"
        confidence = (avg_prob if verdict == "FAKE" else (1.0 - avg_prob)) * 100.0

        return {
            "media_type": "image",
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "fake_probability": round(avg_prob, 4),
            "fake_probability_percent": round(avg_prob * 100, 2),
            "threshold_used": threshold,
            "faces_detected": len(boxes),
            "analyzed_faces": faces_results,
            "dimensions": {"width": w, "height": h}
        }

    def predict_video(self, video_path, num_frames=10, apply_mask=False, generate_heatmap=True, threshold=0.5):
        """
        Analyze a video file by sampling evenly spaced frames.
        Returns aggregate verdict, confidence, and per-frame face inspections.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps if total_frames > 0 else 0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video has 0 frames or is corrupt")

        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        
        frames_data = []
        all_probs = []

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            boxes = self.extract_faces(frame)
            
            if len(boxes) == 0:
                continue

            largest_box = max(boxes, key=lambda b: b[2] * b[3])
            x, y, fw, fh = largest_box
            margin = int(0.15 * max(fw, fh))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)

            face_bgr = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            
            prob, face_url, heatmap_url = self.process_face_crop(face_rgb, apply_mask, generate_heatmap)
            all_probs.append(prob)
            
            timestamp = round(float(f_idx) / fps, 2)
            frames_data.append({
                "frame_index": int(f_idx),
                "timestamp_sec": timestamp,
                "fake_prob": round(prob, 4),
                "fake_prob_percent": round(prob * 100, 2),
                "face_url": face_url,
                "heatmap_url": heatmap_url
            })

        cap.release()

        if len(all_probs) == 0:
            return {
                "media_type": "video",
                "verdict": "INCONCLUSIVE",
                "confidence": 0.0,
                "fake_probability": 0.0,
                "fake_probability_percent": 0.0,
                "threshold_used": threshold,
                "message": "No clear human faces were detected across sampled video frames.",
                "total_video_frames": total_frames,
                "fps": round(fps, 1),
                "duration_seconds": round(duration_sec, 2),
                "sampled_frames_count": len(frame_indices),
                "frames_with_faces": 0,
                "frames_data": []
            }

        avg_prob = float(np.mean(all_probs))
        verdict = "FAKE" if avg_prob >= threshold else "REAL"
        confidence = (avg_prob if verdict == "FAKE" else (1.0 - avg_prob)) * 100.0

        return {
            "media_type": "video",
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "fake_probability": round(avg_prob, 4),
            "fake_probability_percent": round(avg_prob * 100, 2),
            "threshold_used": threshold,
            "total_video_frames": total_frames,
            "fps": round(fps, 1),
            "duration_seconds": round(duration_sec, 2),
            "sampled_frames_count": len(frame_indices),
            "frames_with_faces": len(all_probs),
            "frames_data": frames_data
        }

# Global detector singleton
detector = DeepfakeDetector()
