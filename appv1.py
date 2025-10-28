import streamlit as st
import cv2 as cv
import torch
import json
import base64
import io
import os
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import threading
from vision import Vision
from summary import Summary
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from helper import device, load_json_variable
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

st.set_page_config(
    page_title="VisionSense AI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .alert-critical { 
        background-color: #ff4444; 
        color: white; 
        padding: 10px; 
        border-radius: 5px; 
        margin: 5px 0;
    }
    .alert-high { 
        background-color: #ff8800; 
        color: white; 
        padding: 10px; 
        border-radius: 5px; 
        margin: 5px 0;
    }
    .alert-medium { 
        background-color: #ffaa00; 
        color: white; 
        padding: 10px; 
        border-radius: 5px; 
        margin: 5px 0;
    }
    .alert-none { 
        background-color: #00aa00; 
        color: white; 
        padding: 10px; 
        border-radius: 5px; 
        margin: 5px 0;
    }
    .reasoning-step {
        background-color: #f0f2f6;
        padding: 5px 10px;
        margin: 2px 0;
        border-left: 3px solid #0066cc;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_llava_model():
    with st.spinner("Loading Model"):
        disable_torch_init()
        model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            processor = LlavaNextProcessor.from_pretrained(model_path)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path, 
                dtype=torch.float16, 
                low_cpu_mem_usage=True,
                device_map="auto",
                quantization_config=quantization_config
            )
            
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
            
            return {
                'tokenizer': processor.tokenizer,
                'model': model,
                'image_processor': processor.image_processor,
                'context_len': 4096,
                'processor': processor
            }
        except Exception as e:
            st.error(f"Error loading LLaVA model: {e}")
            return None

def start_camera_stream(vision_system, camera_source, status_callback=None):
    print("Starting camera stream")
    
    llava_model_components = load_llava_model()
    
    if llava_model_components:
        summary_model = Summary(llava_model_components)
        print("Summary model loaded.")

        vision_system = Vision(model_name=load_json_variable("model"), 
                               confidence_threshold=load_json_variable("confidence_threshold"), 
                               stream=load_json_variable("stream"), 
                               summary_model=summary_model)
        
        vision_system.process_viewpoint(camera_source)
        print("Camera stream finished.")
        
    else:
        print("Could not load LLaVA model, exiting camera stream.")
    
        

def get_csv_data():
    try:
        csv_filename = load_json_variable("csv_filename")
        if not os.path.exists(csv_filename):
            return pd.DataFrame()
        
        df = pd.read_csv(csv_filename)
        
        if not df.empty:
            available_columns = df.columns.tolist()
            
            columns_to_keep = ['timestamp', 'summary']
            
            alert_columns = ['alert_status', 'alert_severity', 'alert_type', 'alert_message', 'alert_details_json']
            for col in alert_columns:
                if col in available_columns:
                    columns_to_keep.append(col)
            
            df = df[columns_to_keep].copy()
            
            if 'alert_status' in df.columns:
                df['alert_status'] = df['alert_status'].map({'true': True, 'false': False})
                df['alert_status'] = df['alert_status'].fillna(False)
            
            df = df.sort_values('timestamp', ascending=False)
        
        return df
        
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return pd.DataFrame()

def display_alert_status(row):
    if 'alert_status' in row and row['alert_status']:
        severity = row.get('alert_severity', 'medium')
        message = row.get('alert_message', 'Alert triggered')
        alert_type = row.get('alert_type', 'unknown')
        
        if severity == 'high':
            color = '#ff4444'
            icon = 'Alert'
        elif severity == 'medium':
            color = '#ff8800'
            icon = 'Warning'
        else:
            color = '#ffaa00'
            icon = 'Normal'
        
        st.markdown(f'''
        <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;">
            {icon} <strong>{severity.upper()} ALERT</strong><br>
            Type: {alert_type}<br>
            {message}
        </div>
        ''', unsafe_allow_html=True)
        
        return True
    else:
        st.markdown('''
        <div style="background-color: #00aa00; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;">
            <strong>NORMAL</strong><br>
            No alerts detected
        </div>
        ''', unsafe_allow_html=True)
        return False

def main():
    st.title("🔍 VisionSense AI Dashboard")
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'vision_system' not in st.session_state:
        st.session_state.vision_system = None
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    with st.sidebar:
        st.header("Control Panel")
        
        st.subheader("Camera Control")
        camera_source = st.selectbox("Camera Source", [0, 1, 2], index=0)
        
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("Start Camera", disabled=st.session_state.camera_active):
                st.session_state.camera_active = True
                def start_camera_thread():
                    start_camera_stream(None, camera_source)
                    st.session_state.camera_active = False
                
                camera_thread = threading.Thread(target=start_camera_thread, daemon=True)
                camera_thread.start()
                st.success("Camera window opened! Press 'q' in camera window to stop.")
                st.rerun()
        
        with col_stop:
            if st.button("Stop Camera", disabled=not st.session_state.camera_active):
                st.session_state.camera_active = False
                st.info("Close the camera window or press 'q' to stop streaming.")
                st.rerun()

    tab1, tab2, tab3 = st.tabs(["Live Detection", "Event History", "Memory Viewer"])
    
    with tab1:
        st.header("Live Camera Detection")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("Camera streaming in separate window")
            st.write("Click 'Start Camera' in the sidebar to begin detection.")
        
        with col2:
            st.subheader("Live Status")
            
            df = get_csv_data()
            if not df.empty and st.session_state.camera_active:
                latest_event = df.iloc[0]
                
                st.write("**Latest Summary:**")
                st.write(latest_event['summary'])
                st.write("**Timestamp:**")
                st.write(latest_event['timestamp'])
                
                st.write("**Alert Status:**")
                display_alert_status(latest_event)
                
            else:
                if st.session_state.camera_active:
                    st.info("Waiting for detection data...")
                else:
                    st.info("Start camera to see live status")
            
            if st.session_state.camera_active:
                time.sleep(2)
                st.rerun()
    
    
    
    with tab2:
        st.header("Event History")
        
        df = get_csv_data()
        
        if not df.empty:
            col1, col2 = st.columns(2)
            with col1:
                limit = st.selectbox("Show Last N Events", [10, 25, 50, 100], index=1)
            with col2:
                alert_filter = st.selectbox("Filter by Alert Status", ["All", "Alerts Only", "Normal Only"])
            
            filtered_df = df.copy()
            if alert_filter == "Alerts Only" and 'alert_status' in df.columns:
                filtered_df = filtered_df[filtered_df['alert_status'] == True]
            elif alert_filter == "Normal Only" and 'alert_status' in df.columns:
                filtered_df = filtered_df[filtered_df['alert_status'] == False]
            
            filtered_df = filtered_df.head(limit)
            
            st.subheader("Recent Events")
            
            for idx, row in filtered_df.iterrows():
                alert_indicator = ""
                if 'alert_status' in row and row['alert_status']:
                    severity = row.get('alert_severity', 'medium')
                    if severity == 'high':
                        alert_indicator = "HIGH ALERT - "
                    elif severity == 'medium':
                        alert_indicator = "ALERT - "
                    else:
                        alert_indicator = "Normal` "
                
                with st.expander(f"{alert_indicator}Event {idx} - {row['timestamp']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Timestamp:**", row['timestamp'])
                        st.write("**Summary:**", row['summary'])
                    
                    with col2:
                        st.write("**Alert Status:**")
                        display_alert_status(row)
        else:
            st.info("No events recorded yet.")
        if st.session_state.camera_active:
            time.sleep(2)
            st.rerun()
    
    with tab3:
        st.header("CSV File Viewer")
        
        try:
            csv_filename = load_json_variable("csv_filename")
            print("csv_filename:", csv_filename)
            if os.path.exists(csv_filename):
                df = get_csv_data()
                
                if not df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        if not df.empty:
                            earliest = df['timestamp'].min()
                            latest = df['timestamp'].max()
                            st.metric("Time Range", f"{earliest[:10]} to {latest[:10]}")
                    
                    st.subheader("CSV File Content")
                    st.dataframe(df, use_container_width=True)
                    
                else:
                    st.info("CSV file is empty.")
            else:
                st.info("CSV file not found. Start the camera to generate data.")
            
        except Exception as e:
            st.error(f"Error accessing CSV file: {e}")
        if st.session_state.camera_active:
            time.sleep(2)
            st.rerun()

if __name__ == "__main__":
    main()