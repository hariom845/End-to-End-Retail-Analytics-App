## Real-Time Retail Analytics Dashboard 🛍️

This project is a real-time retail analytics dashboard built with Python, Streamlit, and YOLOv8.
It processes video feeds to provide business insights such as footfall count and customer dwell time in a specified Zone of Interest (ROI).
The dashboard helps retailers understand customer behavior and optimize store layouts.

# ✨ Features
- Interactive Web Dashboard
    Built with Streamlit for an easy-to-use interface.
- Real-Time Person Detection
    Uses YOLOv8 for fast and accurate detection of people in video frames.
- Object Tracking
    Each detected person is assigned a unique ID to track their movement across frames.
- Zone of Interest (ROI)
    Define a polygonal area (e.g., entrance, aisle, promotion zone) to monitor activity.
- Footfall Counting
    Counts the number of unique individuals entering the ROI.
- Dwell Time Analysis
    Calculates how long each customer spends inside the ROI.

# 🛠️ Technology Stack
- Backend: Python
- AI/ML Model: Ultralytics YOLOv8
- Web Framework: Streamlit
- Video Processing: OpenCV
- Data Handling: Pandas, NumPy

# 🚀 Getting Started
1️⃣ Clone the repository
2️⃣ Install dependencies
    - streamlit
    - ultralytics
    - opencv-python
    - numpy
    - pandas
3️⃣ Run the app
4️⃣ Upload a video
    - Use the sidebar to upload a .mp4, .avi, .mov, or .mkv file.
    - Adjust the confidence threshold for detection.
    - View real-time footfall counts and dwell time analysis alongside the video.

# 📊 Example Use Cases
- Measuring foot traffic near promotional displays.
- Monitoring entry/exit zones to understand customer flow.
- Analyzing dwell times in high-value areas of the store.

🔮 Future Enhancements
- Multi-zone analysis
- Heatmaps for movement visualization
- Integration with sales data for deeper insights