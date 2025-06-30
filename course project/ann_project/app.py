import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# MediaPipe Configuration
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global configuration
CONFIG = {
    "min_hand_confidence": 0.7,
    "min_pose_confidence": 0.5,
    "model_complexity": 2,
    "max_faces": 2,
    "refine_landmarks": True,
    "smooth_landmarks": True
}

# 1. PoseDetector Class
class PoseDetector:
    """Advanced pose detection with multi-modal support"""
    
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=CONFIG["model_complexity"],
            smooth_landmarks=CONFIG["smooth_landmarks"],
            min_detection_confidence=CONFIG["min_pose_confidence"]
        )
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=CONFIG["min_hand_confidence"]
        )
        self.face = mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=CONFIG["max_faces"],
            refine_landmarks=CONFIG["refine_landmarks"],
            min_detection_confidence=0.5
        )
        
    def detect(self, image_path):
        """Detects pose with comprehensive error handling"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            
            results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "image_dimensions": {"height": h, "width": w},
                    "detection_settings": CONFIG
                },
                "body": [],
                "hands": {"left": [], "right": []},
                "face": [],
                "gestures": []
            }
            
            # Body detection
            pose_results = self.pose.process(image_rgb)
            if pose_results.pose_landmarks:
                results["body"] = self._process_body_landmarks(pose_results.pose_landmarks, w, h)
            
            # Hand detection
            hands_results = self.hands.process(image_rgb)
            if hands_results.multi_hand_landmarks:
                results["hands"] = self._process_hands(
                    hands_results.multi_hand_landmarks, 
                    hands_results.multi_handedness,
                    w, h
                )
            
            # Face detection
            face_results = self.face.process(image_rgb)
            if face_results.multi_face_landmarks:
                results["face"] = self._process_face(face_results.multi_face_landmarks, w, h)
            
            # Gesture recognition
            results["gestures"] = self._recognize_gestures(results, w)
            
            return results
            
        except Exception as e:
            print(f"Detection Error: {str(e)}")
            return None
    
    def _process_body_landmarks(self, landmarks, w, h):
        """Process body landmarks with anatomical labeling"""
        BODY_LABELS = {
            0: "nose", 11: "left_shoulder", 12: "right_shoulder",
            13: "left_elbow", 14: "right_elbow", 15: "left_wrist",
            16: "right_wrist", 23: "left_hip", 24: "right_hip",
            25: "left_knee", 26: "right_knee", 27: "left_ankle",
            28: "right_ankle"
        }
        return {BODY_LABELS.get(i, f"point_{i}"): [lm.x*w, lm.y*h, lm.z] 
                for i, lm in enumerate(landmarks.landmark)}
    
    def _process_hands(self, hand_landmarks, handedness, w, h):
        """Process hand landmarks with handedness detection"""
        hands = {"left": [], "right": []}
        for idx, hand in enumerate(hand_landmarks):
            hand_type = handedness[idx].classification[0].label.lower()
            hands[hand_type] = [[lm.x*w, lm.y*h, lm.z] for lm in hand.landmark]
        return hands
    
    def _process_face(self, face_landmarks, w, h):
        """Process facial landmarks with key feature extraction"""
        FACE_LABELS = {
            13: "upper_lip", 14: "lower_lip", 
            33: "nose_tip", 152: "chin",
            263: "left_ear", 362: "right_ear"
        }
        return {FACE_LABELS.get(i, f"point_{i}"): [lm.x*w, lm.y*h, lm.z] 
                for i, lm in enumerate(face_landmarks[0].landmark)}
    
    def _recognize_gestures(self, results, w):
        """Basic gesture recognition"""
        gestures = []
        
        # Prayer gesture detection (adjusted threshold for pixel coordinates)
        if (len(results["hands"]["left"]) > 0 and len(results["hands"]["right"]) > 0):
            left_palm = results["hands"]["left"][0]
            right_palm = results["hands"]["right"][0]
            if abs(left_palm[0] - right_palm[0]) < (50 / w) * w:  # Adjusted for pixel coordinates
                gestures.append("prayer_pose")
        
        # Raised hands detection
        if (results["body"].get("left_wrist", [0,0])[1] < results["body"].get("left_shoulder", [0,1])[1] and
            results["body"].get("right_wrist", [0,0])[1] < results["body"].get("right_shoulder", [0,1])[1]):
            gestures.append("hands_raised")
            
        return gestures

# 2. BiomechanicalAnalyzer Class
class BiomechanicalAnalyzer:
    """Advanced biomechanical feature extraction"""
    
    @staticmethod
    def extract_features(pose_data):
        """Comprehensive feature extraction"""
        if not pose_data or not pose_data.get("body"):
            return None
            
        features = {
            "posture": {},
            "joint_angles": {},
            "symmetry": {},
            "facial": {},
            "gestures": pose_data.get("gestures", [])
        }
        
        body = pose_data["body"]
        hands = pose_data["hands"]
        face = pose_data["face"]
        
        # Posture classification
        features["posture"] = BiomechanicalAnalyzer._classify_posture(body)
        
        # Joint angle analysis
        features["joint_angles"] = {
            "right_elbow": BiomechanicalAnalyzer._calculate_angle(
                body["right_shoulder"], body["right_elbow"], body["right_wrist"]),
            "left_elbow": BiomechanicalAnalyzer._calculate_angle(
                body["left_shoulder"], body["left_elbow"], body["left_wrist"]),
            "right_knee": BiomechanicalAnalyzer._calculate_angle(
                body["right_hip"], body["right_knee"], body["right_ankle"]),
            "left_knee": BiomechanicalAnalyzer._calculate_angle(
                body["left_hip"], body["left_knee"], body["left_ankle"])
        }
        
        # Body symmetry
        features["symmetry"] = {
            "shoulder_level": abs(body["left_shoulder"][1] - body["right_shoulder"][1]),
            "hip_level": abs(body["left_hip"][1] - body["right_hip"][1]),
            "arm_angle_diff": abs(features["joint_angles"]["right_elbow"] - 
                                 features["joint_angles"]["left_elbow"]),
            "nose_hip_diff": body["nose"][1] - body["right_hip"][1]  # Added for FitnessCoach
        }
        
        # Facial analysis
        if face:
            features["facial"] = {
                "smile_intensity": face["lower_lip"][1] - face["upper_lip"][1],
                "eye_aspect_ratio": BiomechanicalAnalyzer._calculate_eye_aspect_ratio(face)
            }
        
        return features
    
    @staticmethod
    def _calculate_angle(a, b, c):
        """Calculate joint angle in degrees"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    
    @staticmethod
    def _classify_posture(body):
        """Advanced posture classification"""
        posture = {}
        
        # Vertical alignment
        shoulder_hip_ratio = abs(body["right_shoulder"][0] - body["right_hip"][0])
        if shoulder_hip_ratio < 0.05:
            posture["alignment"] = "excellent"
        elif shoulder_hip_ratio < 0.1:
            posture["alignment"] = "good"
        else:
            posture["alignment"] = "poor"
        
        # Forward lean
        nose_hip_diff = body["nose"][1] - body["right_hip"][1]
        if nose_hip_diff < -0.1:
            posture["lean"] = "forward"
        elif nose_hip_diff > 0.1:
            posture["lean"] = "backward"
        else:
            posture["lean"] = "neutral"
            
        return posture
    
    @staticmethod
    def _calculate_eye_aspect_ratio(face):
        """Calculate eye aspect ratio for blink detection"""
        return 0.3  # Placeholder

# 3. PoseCaptionGenerator Class
class PoseCaptionGenerator:
    """Advanced natural language description system"""
    
    TEMPLATES = {
        "posture": {
            "excellent": "perfectly aligned",
            "good": "well-aligned",
            "poor": "misaligned"
        },
        "lean": {
            "forward": "leaning forward",
            "backward": "leaning backward",
            "neutral": "vertically straight"
        },
        "angles": {
            "elbow": {
                (0, 45): "sharply bent",
                (45, 90): "bent at right angle",
                (90, 135): "slightly bent",
                (135, 180): "fully extended"
            },
            "knee": {
                (0, 60): "deeply bent",
                (60, 100): "moderately bent",
                (100, 140): "slightly bent",
                (140, 180): "straight"
            }
        }
    }
    
    @classmethod
    def generate_caption(cls, features):
        """Generate comprehensive pose description"""
        if not features:
            return "No pose detected"
            
        caption_parts = []
        
        # Posture description
        posture = features["posture"]
        caption_parts.append(f"Person is standing with {cls.TEMPLATES['posture'][posture['alignment']]} posture, "
                           f"{cls.TEMPLATES['lean'][posture['lean']]}")
        
        # Arm description
        arm_desc = []
        for side in ["right", "left"]:
            angle = features["joint_angles"][f"{side}_elbow"]
            desc = cls._describe_angle(angle, "elbow")
            arm_desc.append(f"{side} arm {desc}")
        if arm_desc:
            caption_parts.append(" with " + " and ".join(arm_desc))
        
        # Leg description
        leg_desc = []
        for side in ["right", "left"]:
            angle = features["joint_angles"][f"{side}_knee"]
            desc = cls._describe_angle(angle, "knee")
            leg_desc.append(f"{side} leg {desc}")
        if leg_desc:
            caption_parts.append(", " + " and ".join(leg_desc))
        
        # Facial expression
        if features.get("facial"):
            smile = features["facial"]["smile_intensity"]
            if smile > 0.05:
                caption_parts.append(", smiling confidently")
            elif smile > 0.02:
                caption_parts.append(", with a slight smile")
            else:
                caption_parts.append(", with neutral expression")
        
        # Gestures
        if features["gestures"]:
            caption_parts.append(f". Gestures detected: {', '.join(features['gestures'])}")
        
        return "".join(caption_parts) + "."
    
    @classmethod
    def _describe_angle(cls, angle, joint_type):
        """Get natural language description for joint angle"""
        for (min_val, max_val), desc in cls.TEMPLATES["angles"][joint_type].items():
            if min_val <= angle < max_val:
                return desc
        return "at unknown angle"

    @classmethod
    def generate_simple_caption(cls, features):
        """Generate a shorter version of the caption"""
        if not features:
            return "No pose detected"
        
        posture = features["posture"]
        caption = f"Person with {posture['alignment']} posture, {posture['lean']}"
        
        # Add main angle info
        avg_elbow = (features["joint_angles"]["right_elbow"] + features["joint_angles"]["left_elbow"]) / 2
        avg_knee = (features["joint_angles"]["right_knee"] + features["joint_angles"]["left_knee"]) / 2
        
        caption += f". Arms at ~{int(avg_elbow)}°, legs at ~{int(avg_knee)}°"
        
        if features["gestures"]:
            caption += f" ({', '.join(features['gestures'])})"
            
        return caption

# 4. FitnessCoach Class
class FitnessCoach:
    """Advanced fitness feedback system"""
    
    STANDARDS = {
        "squat": {
            "optimal_knee_angle": (85, 100),
            "optimal_torso_lean": (-0.05, 0.05)
        },
        "pushup": {
            "optimal_elbow_angle": (75, 90),
            "shoulder_alignment_tolerance": 0.03
        }
    }
    
    @classmethod
    def analyze_pose(cls, features, exercise_type="squat"):
        """Generate professional fitness feedback"""
        if not features:
            return ["No pose detected"]
            
        feedback = []
        standards = cls.STANDARDS.get(exercise_type, {})
        
        if exercise_type == "squat":
            # Knee angle analysis
            knee_angle = (features["joint_angles"]["right_knee"] + 
                         features["joint_angles"]["left_knee"]) / 2
            optimal = standards["optimal_knee_angle"]
            
            if optimal[0] <= knee_angle <= optimal[1]:
                feedback.append(f"Perfect squat depth ({int(knee_angle)}°)")
            elif knee_angle < optimal[0]:
                feedback.append(f"Too deep! Reduce squat depth (current: {int(knee_angle)}°, target: {optimal[0]}-{optimal[1]}°)")
            else:
                feedback.append(f"Not deep enough (current: {int(knee_angle)}°, target: {optimal[0]}-{optimal[1]}°)")
            
            # Torso lean analysis
            nose_hip_diff = features["symmetry"]["nose_hip_diff"]
            if not (standards["optimal_torso_lean"][0] <= nose_hip_diff <= standards["optimal_torso_lean"][1]):
                feedback.append(f"Keep torso more vertical (currently {'leaning forward' if nose_hip_diff < 0 else 'leaning backward'})")
        
        elif exercise_type == "pushup":
            # Elbow angle analysis
            elbow_angle = (features["joint_angles"]["right_elbow"] + 
                          features["joint_angles"]["left_elbow"]) / 2
            optimal = standards["optimal_elbow_angle"]
            
            if optimal[0] <= elbow_angle <= optimal[1]:
                feedback.append(f"Good pushup form ({int(elbow_angle)}°)")
            else:
                feedback.append(f"Adjust elbow angle (current: {int(elbow_angle)}°, target: {optimal[0]}-{optimal[1]}°)")
            
            # Shoulder symmetry
            if features["symmetry"]["shoulder_level"] > standards["shoulder_alignment_tolerance"]:
                feedback.append("Keep shoulders level during movement")
        
        # General feedback
        if features["symmetry"]["arm_angle_diff"] > 15:
            feedback.append("Maintain symmetrical arm positioning")
        
        if not feedback:
            feedback.append("Good form detected!")
            
        return feedback

# 5. YogaInstructor Class
class YogaInstructor:
    """Professional yoga pose analysis"""
    
    POSES = {
        "tadasana": {
            "description": "Mountain Pose",
            "knee_angle": (170, 180),
            "elbow_angle": (160, 180),
            "shoulder_alignment": 0.03
        },
        "utkatasana": {
            "description": "Chair Pose",
            "knee_angle": (90, 120),
            "elbow_angle": (160, 180),
            "shoulder_alignment": 0.05
        }
    }
    
    @classmethod
    def analyze_pose(cls, features):
        """Identify and assess yoga poses"""
        if not features:
            return "No pose detected"
            
        # Try to match known poses
        for pose_name, standards in cls.POSES.items():
            knee_match = (standards["knee_angle"][0] <= features["joint_angles"]["right_knee"] <= standards["knee_angle"][1])
            elbow_match = (standards["elbow_angle"][0] <= features["joint_angles"]["right_elbow"] <= standards["elbow_angle"][1])
            
            if knee_match and elbow_match:
                feedback = [f"Detected {standards['description']}"]
                
                # Alignment check
                if features["symmetry"]["shoulder_level"] > standards["shoulder_alignment"]:
                    feedback.append("Improve shoulder alignment")
                
                if features["symmetry"]["hip_level"] > standards["shoulder_alignment"]:
                    feedback.append("Level your hips")
                
                return ". ".join(feedback) + "."
        
        return "Pose not recognized. Focus on alignment and try again."

# 6. PoseDatasetBuilder Class
class PoseDatasetBuilder:
    """Tool for creating and managing pose-caption datasets"""
    
    def __init__(self, dataset_dir="pose_dataset"):
        self.dataset_dir = dataset_dir
        os.makedirs(dataset_dir, exist_ok=True)
        self.metadata_file = os.path.join(dataset_dir, "metadata.json")
        self.annotations_file = os.path.join(dataset_dir, "annotations.csv")
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "description": "Pose2Caption Dataset",
                "version": "1.0",
                "categories": {
                    "yoga": ["tadasana", "utkatasana", "virabhadrasana"],
                    "fitness": ["squat", "lunge", "pushup"],
                    "daily": ["standing", "sitting", "walking"]
                },
                "stats": {"total_images": 0}
            }
        
        if os.path.exists(self.annotations_file):
            self.df = pd.read_csv(self.annotations_file)
        else:
            self.df = pd.DataFrame(columns=[
                "image_path", "category", "pose_type", 
                "caption", "features", "validation_status"
            ])
    
    def add_image(self, image_path, category, pose_type, caption=None):
        """Add a new image to the dataset"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Create unique filename
        base_name = f"{category}_{pose_type}_{self.metadata['stats']['total_images']}.jpg"
        new_path = os.path.join(self.dataset_dir, base_name)
        
        # Copy/resize image
        img = Image.open(image_path)
        img = img.resize((640, 480))  # Standardize size
        img.save(new_path)
        
        # Detect pose features
        detector = PoseDetector()
        pose_data = detector.detect(new_path)
        features = BiomechanicalAnalyzer().extract_features(pose_data) if pose_data else None
        
        # Add to dataframe
        new_entry = {
            "image_path": new_path,
            "category": category,
            "pose_type": pose_type,
            "caption": caption,
            "features": json.dumps(features) if features else None,
            "validation_status": "pending"
        }
        
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
        self.metadata["stats"]["total_images"] += 1
        
        # Save updates
        self._save()
        
        return new_path
    
    def generate_captions(self):
        """Generate automatic captions for all images"""
        for idx, row in self.df.iterrows():
            if pd.isna(row['caption']) and row['features']:
                features = json.loads(row['features'])
                self.df.at[idx, 'caption'] = PoseCaptionGenerator().generate_caption(features)
        self._save()
    
    def _save(self):
        """Save dataset to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        self.df.to_csv(self.annotations_file, index=False)
    
    def get_dataset(self):
        """Return the complete dataset"""
        return self.df
    
    def analyze_dataset(self):
        """Generate dataset statistics and visualizations"""
        st.write("### Dataset Statistics")
        st.write(f"Total Images: {self.metadata['stats']['total_images']}")
        
        # Category distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=self.df, x='category', ax=ax)
        ax.set_title("Pose Category Distribution")
        st.pyplot(fig)
        plt.close(fig)  # Close the figure to free memory
        
        # Validation status
        if 'validation_status' in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=self.df, x='validation_status', ax=ax)
            ax.set_title("Annotation Validation Status")
            st.pyplot(fig)
            plt.close(fig)  # Close the figure to free memory

# Streamlit App Code
# Configure Streamlit
st.set_page_config(page_title="Pose2Caption", layout="wide")
st.title("Pose2Caption: Advanced Pose Analysis System")

# Initialize systems
detector = PoseDetector()
analyzer = BiomechanicalAnalyzer()
generator = PoseCaptionGenerator()
fitness_coach = FitnessCoach()
yoga_instructor = YogaInstructor()
dataset_builder = PoseDatasetBuilder()

# Sidebar controls
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.7)
exercise_type = st.sidebar.selectbox("Exercise Type", ["squat", "pushup", "yoga"])

# Main interface
upload_tab, dataset_tab = st.tabs(["Upload Image", "Dataset Management"])

with upload_tab:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Save to temp file
        img_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                img_path = tmp_file.name
            
            # Process image
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_path, caption="Uploaded Image", use_column_width=True)
                
                # Detect pose
                pose_data = detector.detect(img_path)
                if pose_data:
                    # Visualize pose
                    image = cv2.imread(img_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pose = detector.pose.process(image_rgb)
                    if pose.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image, pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        st.image(image, caption="Pose Detection", channels="BGR", use_column_width=True)
            
            with col2:
                if pose_data:
                    features = analyzer.extract_features(pose_data)
                    if features:
                        # Generate outputs
                        st.subheader("Analysis Results")
                        
                        st.markdown("### Generated Caption")
                        simple_caption = generator.generate_simple_caption(features)
                        detailed_caption = generator.generate_caption(features)
                        
                        st.info(simple_caption)
                        with st.expander("Detailed Description"):
                            st.write(detailed_caption)
                        
                        st.markdown("### Fitness Feedback")
                        feedback = fitness_coach.analyze_pose(features, exercise_type)
                        for item in feedback:
                            if "adjust" in item.lower() or "not" in item.lower():
                                st.warning(item)
                            else:
                                st.success(item)
                        
                        st.markdown("### Yoga Assessment")
                        yoga_feedback = yoga_instructor.analyze_pose(features)
                        st.info(yoga_feedback)
        
        finally:
            # Clean up
            if img_path and os.path.exists(img_path):
                os.unlink(img_path)

with dataset_tab:
    st.header("Pose Dataset Management")
    
    # Add new pose to dataset
    st.subheader("Add New Pose")
    new_category = st.selectbox("Category", ["yoga", "fitness", "daily"])
    new_pose_type = st.text_input("Pose Type (e.g., 'tadasana')")
    new_caption = st.text_area("Professional Caption")
    new_image = st.file_uploader("Upload Pose Image", type=["jpg", "png", "jpeg"], key="dataset_uploader")
    
    if st.button("Add to Dataset") and new_image and new_pose_type:
        img_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(new_image.getvalue())
                img_path = tmp_file.name
            
            dataset_builder.add_image(img_path, new_category, new_pose_type, new_caption)
            st.success("Pose added to dataset!")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        finally:
            if img_path and os.path.exists(img_path):
                os.unlink(img_path)
    
    # View dataset
    st.subheader("Dataset Statistics")
    if st.button("Refresh Dataset"):
        dataset = dataset_builder.get_dataset()
        if not dataset.empty:
            st.dataframe(dataset)
            
            # Show stats
            st.write(f"Total Poses: {len(dataset)}")
            dataset_builder.analyze_dataset()
        else:
            st.info("Dataset is empty. Add poses to see statistics.")