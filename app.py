import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import tempfile

#Model Loading
@st.cache_resource
def load_yolo_model():
    """
    Loads the YOLOv8 model from the 'ultralytics' library.
    Using st.cache_resource to load the model only once.
    """
    try:
        model = YOLO('yolov8n.pt')  # Using the smallest, fastest model
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

#Main Application Logic
def main():
    """
    The main function that sets up the Streamlit interface and runs the
    video processing logic.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Retail Analytics Dashboard",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Real-Time Retail Analytics Dashboard 🛍️")
    st.markdown(
        "Upload a video to analyze **Footfall Count** and **Dwell Time** in a "
        "pre-defined 'Zone of Interest'. The application uses the **YOLOv8** model for "
        "real-time person detection and tracking."
    )
    st.markdown("---")

    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader(
            "Upload a video file", type=['mp4', 'mov', 'avi', 'mkv']
        )
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.05
        )
        st.info(
            "A lower confidence threshold may increase detections but also "
            "introduce more false positives."
        )

    # Main Area for Displaying Video and Metrics
    col1, col2 = st.columns([3, 2]) # Ratio for video feed vs metrics

    with col1:
        st.header("🎥 Video Feed")
        video_placeholder = st.empty()

    with col2:
        st.header("📊 Real-Time Metrics")
        footfall_kpi = st.empty()
        st.subheader("Dwell Time Analysis")
        dwell_time_table = st.empty()


    # --- Video Processing Logic ---
    if uploaded_file is not None:
        model = load_yolo_model()
        if model is None:
            return # Stop if model failed to load

        # Use a temporary file to handle the video stream with OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.getvalue())
            video_path = tfile.name

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                return

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define the 'Zone of Interest' as a polygon covering the bottom half
            zone_polygon = np.array([
                [0, height // 2],
                [width, height // 2],
                [width, height],
                [0, height]
            ], np.int32)
            zone_polygon_reshaped = zone_polygon.reshape((-1, 1, 2))

            # --- Data Storage for Analytics ---
            # {track_id: [center_x, center_y]}
            track_history = defaultdict(list)
            # {track_id: start_frame} - To calculate current session duration
            dwell_time_start = {}
            # {track_id: total_frames} - Accumulates dwell time across multiple entries
            total_dwell_time = defaultdict(int)
            # Set of unique track_ids that have entered the zone
            footfall_set = set()
            # Set of track_ids currently inside the zone
            people_in_zone = set()

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.write("Video processing finished.")
                    break
                
                frame_count += 1

                # Perform object detection and tracking on the frame
                results = model.track(
                    frame, persist=True, classes=[0], conf=confidence_threshold, verbose=False
                )

                # Use the annotated frame from the model results
                annotated_frame = results[0].plot()
                # Draw the zone of interest on the frame
                cv2.polylines(annotated_frame, [zone_polygon_reshaped], isClosed=True, color=(255, 165, 0), thickness=2)
                cv2.putText(annotated_frame, "Zone of Interest", (zone_polygon[0][0] + 10, zone_polygon[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    current_people_this_frame = set()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        # Use the bottom center of the bounding box as the anchor point
                        point = (int(x), int(y + h / 2))
                        
                        # Check if the person's anchor point is inside the zone
                        is_inside = cv2.pointPolygonTest(zone_polygon, point, False) >= 0

                        if is_inside:
                            current_people_this_frame.add(track_id)
                            # This is the first time this person has entered the zone
                            footfall_set.add(track_id)

                            # If person just entered, record their entry frame number
                            if track_id not in people_in_zone:
                                dwell_time_start[track_id] = frame_count

                    # --- Dwell Time Calculation ---
                    # Identify people who have left the zone in this frame
                    people_who_left = people_in_zone - current_people_this_frame
                    for track_id in people_who_left:
                        if track_id in dwell_time_start:
                            duration_frames = frame_count - dwell_time_start.pop(track_id)
                            total_dwell_time[track_id] += duration_frames

                    # Update the master set of people currently in the zone
                    people_in_zone = current_people_this_frame

                # --- Update Dashboard Metrics ---
                footfall_kpi.metric("Total Footfall Count", value=f"🚶 {len(footfall_set)}")
                
                # Prepare data for the dwell time table
                dwell_data = []
                all_tracked_ids = set(total_dwell_time.keys()) | people_in_zone
                
                for track_id in sorted(list(all_tracked_ids)):
                    status = "Inside Zone" if track_id in people_in_zone else "Left Zone"
                    
                    # Get accumulated time for people who have left
                    total_frames = total_dwell_time.get(track_id, 0)
                    
                    # If person is currently inside, add their current session time
                    if track_id in dwell_time_start:
                        current_session_frames = frame_count - dwell_time_start[track_id]
                        total_frames += current_session_frames

                    total_seconds = total_frames / fps if fps > 0 else 0
                    
                    dwell_data.append({
                        "Person ID": track_id,
                        "Dwell Time (s)": f"{total_seconds:.2f}",
                        "Status": status
                    })
                
                # Display the dwell time table
                if dwell_data:
                    df = pd.DataFrame(dwell_data).set_index("Person ID")
                    dwell_time_table.dataframe(df)
                else:
                    dwell_time_table.empty()

                # --- Display Video ---
                # THIS IS THE CORRECTED LINE
                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        finally:
            cap.release() # Ensure video capture is released

    else:
        st.info("Please upload a video file to begin analysis.")


if __name__ == "__main__":
    main()
