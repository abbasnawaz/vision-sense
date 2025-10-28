import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from helper import load_json_variable

class AlertSystem:
    
    def __init__(self):
        self.night_start = load_json_variable('night_start')
        self.night_end = load_json_variable('night_end')
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    
    def is_night_time(self, timestamp: str) -> bool:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            
            if self.night_start > self.night_end:
                return hour >= self.night_start or hour < self.night_end
            else:
                return self.night_start <= hour < self.night_end
                
        except Exception as e:
            print(f"Error parsing timestamp {timestamp}: {e}")
            return False
    
    def count_objects_by_type(self, detections: List[Dict]) -> Dict[str, int]:
        counts = {}
        for detection in detections:
            class_name = detection.get('class_name', '').lower()
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def analyze_detections(self, detections: List[Dict], timestamp: str) -> Dict[str, Any]:
        if not detections:
            return {
                'should_alert': False,
                'severity': 'none',
                'alert_type': 'none',
                'message': 'No detections',
                'details': {},
                'timestamp': timestamp
            }
        
        object_counts = self.count_objects_by_type(detections)
        person_count = object_counts.get('person', 0)
        is_night = self.is_night_time(timestamp)
        
        vehicle_count = sum(object_counts.get(vehicle, 0) for vehicle in self.vehicle_classes)
        
        alerts = []
        severity = 'none'
        alert_type = 'none'
        
        if is_night:
            if person_count > 3:
                alerts.append({
                    'type': 'night_person_count',
                    'message': f'Night time: {person_count} persons detected (threshold: 3)',
                    'severity': 'high'
                })
                severity = 'high'
                alert_type = 'night_intrusion'
        
        else:
            if person_count > 7:
                alerts.append({
                    'type': 'day_person_count', 
                    'message': f'Day time: {person_count} persons detected (threshold: 7)',
                    'severity': 'medium'
                })
                severity = 'medium'
                alert_type = 'crowd_detection'
        
        if vehicle_count > 5:
            alerts.append({
                'type': 'vehicle_count',
                'message': f'{vehicle_count} vehicles detected (threshold: 5)',
                'severity': 'medium'
            })
            if severity == 'none':
                severity = 'medium'
            if alert_type == 'none':
                alert_type = 'traffic'
        
        should_alert = len(alerts) > 0
        
        if should_alert:
            time_period = 'Night' if is_night else 'Day'
            summary = f"{time_period} alert: {len(alerts)} threshold(s) exceeded"
        else:
            summary = f"Normal activity - {person_count} persons, {vehicle_count} vehicles"
        
        return {
            'should_alert': should_alert,
            'severity': severity,
            'alert_type': alert_type,
            'summary': summary,
            'message': summary,
            'alerts': alerts,
            'details': {
                'is_night': is_night,
                'person_count': person_count,
                'vehicle_count': vehicle_count,
                'object_counts': object_counts,
                'total_detections': len(detections)
            },
            'timestamp': timestamp
        }
    
    def format_alert_message(self, alert_data: Dict) -> str:
        if not alert_data.get('should_alert', False):
            return alert_data.get('summary', 'No alerts')
        
        message = f"{alert_data.get('summary', 'Alert')}\n"
        
        for alert in alert_data.get('alerts', []):
            message += f"{alert['message']}\n"
        
        details = alert_data.get('details', {})
        message += f"\nDetails: {details['person_count']} persons, {details['vehicle_count']} vehicles"
        
        return message.strip()
