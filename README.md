# VisionSense - Intelligent Security Monitoring System

> **An AI-powered security monitoring system that combines computer vision with intelligent reasoning to provide real-time threat detection and analysis.**

VisionSense is more than just object detection - it's an intelligent agent that *thinks* about what it sees, learns from patterns, and makes smart decisions about when to raise security alerts.

## 🌟 What Makes VisionSense Special?

Instead of just detecting objects, VisionSense:
- **Thinks contextually** - A person at 2 PM vs 2 AM means different things
- **Remembers patterns** - Learns what "normal" looks like in your environment  
- **Explains decisions** - Shows you *why* it raised an alert
- **Speaks naturally** - Generates human-readable summaries like "3 people detected near entrance after hours - possible intrusion"

## 🏗️ System Architecture

VisionSense follows a **modular, intelligent agent architecture** designed for real-world security applications:

```
┌─────────────────────────────────────────────────────────────┐
│                    🎯 VisionSense Core                      │
├─────────────────────────────────────────────────────────────┤
│  📹 Vision Module     │  🧠 AI Reasoning     │  🚨 Alerts   │
│  • YOLOv8 Detection  │  • LLaVA Summaries   │  • Rule Engine│
│  • Object Tracking   │  • Context Analysis  │  • Severity   │
│  • Confidence Score  │  • Pattern Learning  │  • Logging    │
├─────────────────────────────────────────────────────────────┤
│  💾 Memory System    │  📊 Web Dashboard    │  ⚙️ Config    │
│  • CSV Logging       │  • Live Monitoring   │  • Time Rules │
│  • Event History     │  • Analytics View    │  • Thresholds │
│  • Pattern Storage   │  • Alert Management  │  • Flexibility│
└─────────────────────────────────────────────────────────────┘
```

## 🧩 Core Components

### 1. **Vision Engine** (`vision.py`)
The eyes of the system - powered by YOLOv8 for lightning-fast object detection.

**Why YOLOv8?** 
- ⚡ Real-time performance (30+ FPS)
- 🎯 High accuracy for security-relevant objects
- 🔧 Easy to customize and deploy

```python
# Detects objects with confidence scores
detections = vision.detect_objects(frame)
# Output: [{"class_name": "person", "confidence": 0.89, "bbox": [...]}]
```

### 2. **AI Reasoning Module** (`summary.py`)
The brain - uses LLaVA (Large Language and Vision Assistant) to understand *context*.

**Why LLaVA?**
- 🧠 Understands both images AND text
- 💬 Generates natural language explanations
- 🔍 Provides contextual analysis beyond simple detection

```python
# Generates intelligent summaries
summary = "2 people detected in parking area during night hours - unusual activity"
```

### 3. **Alert Intelligence** (`alert_system.py`)
The decision maker - applies configurable rules with explainable reasoning.

**Smart Alert Logic:**
```python
# Night time (22:00-06:00): Alert if >3 people
# Day time (06:00-22:00): Alert if >7 people  
# Any time: Alert if >5 vehicles
```

**Design Philosophy:** *Reduce false positives while catching real threats*

### 4. **Memory System** 
Learns and remembers patterns for intelligent decision-making.

- **CSV Logging** - Fast, simple, human-readable
- **Pattern Recognition** - Compares current vs historical activity
- **Context Preservation** - Maintains temporal awareness

### 5. **Web Dashboard** (`streamlit_dashboard.py`)
Professional monitoring interface for security personnel.

**Real-time Features:**
- 🎥 Live camera feed with AI overlays
- 📊 Analytics and trend visualization  
- 📋 Searchable event history
- 🚨 Alert management and filtering

## 🎨 Design Choices Explained

### **Why This Architecture?**

1. **Modularity** - Each component has a single responsibility
2. **Scalability** - Easy to add new detection models or alert rules
3. **Explainability** - Security teams need to understand *why* alerts fire
4. **Performance** - Optimized for real-time operation
5. **Flexibility** - Configurable for different environments

### **Why CSV over Database?**
- ✅ **Human-readable** - Security teams can open files directly
- ✅ **Fast writes** - No database overhead for high-frequency logging
- ✅ **Simple backup** - Just copy files
- ✅ **Tool compatibility** - Works with Excel, Python, R, etc.

### **Why Hybrid AI Approach?**
- **Rules Engine** - Fast, explainable, deterministic
- **LLM Summaries** - Natural language, contextual understanding
- **Best of both worlds** - Speed + intelligence

## 🚀 Quick Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# CUDA (optional, for GPU acceleration)
nvidia-smi
```

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd VisionSense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Edit `config.json` for your environment:
```json
{
    "model": "yolov8n.pt",
    "confidence_threshold": 0.6,
    "night_start": 22,
    "night_end": 6,
    "csv_filename": "vision_summaries.csv"
}
```

### Quick Start
```bash
# Test with single image
python app.py

# Launch web dashboard
streamlit run streamlit_dashboard.py

# Access dashboard at: http://localhost:8501
```

## 📱 Using the Dashboard

### 1. **Start Monitoring**
- Click "▶️ Start Camera" in sidebar
- Select camera source (0 for webcam, or IP camera URL)
- Adjust confidence threshold as needed

### 2. **Monitor Live Feed**
- **Green boxes** = Normal activity
- **Yellow boxes** = Medium alerts  
- **Red boxes** = High priority alerts
- Real-time reasoning displayed below feed

### 3. **Review Analytics**
- Switch to "Analytics" tab for trends
- Check alert frequency and patterns
- Identify peak activity times

### 4. **Investigate Events**
- Use "Event History" for detailed review
- Filter by alert status or time range
- Export data for further analysis

## ⚙️ Customization

### Adding New Alert Rules
```python
# In alert_system.py
def analyze_custom_scenario(self, detections, timestamp):
    # Your custom logic here
    if custom_condition:
        return {
            'alert': True,
            'message': 'Custom alert triggered',
            'severity': 'high'
        }
```

### Configuring for Your Environment
```python
# Adjust thresholds in config.json
{
    "person_threshold_day": 7,    # Max people during day
    "person_threshold_night": 3,  # Max people during night
    "vehicle_threshold": 5,       # Max vehicles anytime
    "confidence_threshold": 0.6   # Detection confidence
}
```

## 🔧 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Test camera access
python -c "import cv2; print(cv2.VideoCapture(0).read())"
```

**Model loading slowly:**
```bash
# Use lighter model for faster startup
# In config.json: "model": "yolov8n.pt"  # nano version
```

**High CPU usage:**
```bash
# Reduce frame rate or resolution in vision.py
cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS
```
