# VisionSense 

- **Thinks contextually** - A person at 2 PM vs 2 AM means different things
- **Remembers patterns** - Learns what "normal" looks like in your environment  
- **Explains decisions** - Shows you *why* it raised an alert
- **Speaks naturally** - Generates human-readable summaries like "3 people detected near entrance after hours - possible intrusion"

## 🏗️ System Architecture

VisionSense follows a **modular, intelligent agent architecture** designed for real-world security applications:



## 🧩 Core Components

### 1. **Vision Engine** (`vision.py`)
Powered by YOLOv8 for lightning-fast object detection.

*

```python
# Detects objects with confidence scores
detections = vision.detect_objects(frame)
# Output: [{"class_name": "person", "confidence": 0.89, "bbox": [...]}]
```

### 2. **AI Reasoning Module** (`summary.py`)
uses LLaVA (Large Language and Vision Assistant) to understand *context*.

**Why LLaVA?**
- Understands both images AND text
- Generates natural language explanations
- Provides contextual analysis beyond simple detection

```python
# Generates intelligent summaries
summary = "2 people detected in parking area during night hours - unusual activity"
```

### 3. **Alert Intelligence** (`alert_system.py`)
Applies configurable rules with explainable reasoning.

**Smart Alert Logic:**
```python
# Night time (22:00-06:00): Alert if >3 people
# Day time (06:00-22:00): Alert if >7 people  
# Any time: Alert if >5 vehicles
```

### 4. **Memory System** 
Learns and remembers patterns for intelligent decision-making.

- **CSV Logging**
- **Pattern Recognition**
- **Context Preservation**

### 5. **Web Dashboard** (`app.py`)
Professional monitoring interface for security personnel.

### **Why This Architecture?**

1. Each component has a single responsibility
2. Easy to add new detection models or alert rules
3. Optimized for real-time operation
4. Configurable for different environments

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
