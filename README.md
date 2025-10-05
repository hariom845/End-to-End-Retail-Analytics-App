Real-Time Retail Analytics Dashboard 🛍️
This project is a real-time retail analytics application built with Python, Streamlit, and YOLOv8. It analyzes a video feed to provide key business metrics like Footfall Count and customer Dwell Time within a specified area, helping businesses understand customer behavior and optimize store layouts.

The interactive web dashboard allows users to upload their own video files and see the analysis happen in real time.

working link - https://retail-analytics-app-njhvl3kcqjef34pr9jmrpk.streamlit.app/

✨ Features
Interactive Web Dashboard: A user-friendly interface built with Streamlit.

Real-Time Person Detection: Utilizes the powerful YOLOv8 model for accurate and fast person detection.

Object Tracking: Assigns a unique ID to each person to track their movement across frames.

Zone of Interest (ROI): Define a custom polygonal zone to monitor activity in a specific area (e.g., near a promotion, an entrance, or a specific aisle).

Footfall Counting: Counts the total number of unique individuals who enter the Zone of Interest.

Dwell Time Analysis: Calculates the amount of time each person spends inside the zone, providing insights into customer engagement.

🛠️ Technology Stack
Backend: Python

AI/ML Model: Ultralytics YOLOv8

Web Framework: Streamlit

Video Processing: OpenCV

Data Handling: Pandas, NumPy

🚀 How to Run Locally
Clone the repository:

Bash

git clone https://github.com/your-username/retail-analytics-app.git
cd retail-analytics-app
Install the dependencies:

Bash

pip install -r requirements.txt
Run the Streamlit application:

Bash

streamlit run app.py
