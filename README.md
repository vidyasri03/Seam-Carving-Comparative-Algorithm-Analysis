# 🖼️ Seam Carving – Comparative Algorithm Analysis

A Python-based image processing project that implements and compares multiple Seam Carving algorithms for content-aware image resizing. The project provides an interactive Streamlit application for energy-map analysis, seam detection, seam removal, and algorithm comparison.

## 📌 Project Overview

Seam Carving is a content-aware image resizing technique that removes low-energy paths, called seams, from an image while preserving important visual content.

This project implements and analyzes different approaches for identifying optimal seams and studies their performance, computational efficiency, and image-quality impact.

The project includes an interactive web application where users can:

- Upload images
- Compare energy maps
- Detect optimal seams
- Visualize seam paths
- Remove multiple seams
- Compare different algorithms
- Analyze image-processing results

## ✨ Features

### 🔹 Energy Map Comparison

The application compares two energy-map techniques:

- Sobel Energy Map
- Laplacian Energy Map

The application displays the generated energy maps along with their total energy values and a comparison chart.

### 🔹 Seam Detection

The system identifies a low-energy seam from the image and visualizes it.

It provides:

- Original image
- Image with detected seam
- Energy map
- Seam path
- Seam length
- Total seam energy

### 🔹 Seam Removal

Users can select the number of seams to remove from the image and observe the resulting content-aware resizing.

### 🔹 Algorithm Comparison

Multiple approaches are implemented and analyzed:

- Greedy Algorithm
- Recursive Algorithm
- Dynamic Programming
- Graph-based approach using Dijkstra's Algorithm

## 🧠 Algorithms

| Algorithm | Approach | Main Idea |
|---|---|---|
| Greedy | Local Optimization | Selects the lowest-energy neighboring pixel at each step |
| Recursive | Recursive Search | Explores possible seam paths recursively |
| Dynamic Programming | Optimal Substructure | Computes the minimum-energy seam efficiently |
| Dijkstra | Graph-based | Models pixels as graph nodes and finds a minimum-cost path |

## 📊 Performance Analysis

The project evaluates the algorithms using:

- Execution Time
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Seam Energy
- Image Quality

These metrics are used to study the trade-off between algorithm efficiency and image preservation.

## 🖥️ Interactive Application

The project contains three main sections:

### 1. 🔍 Energy Map Comparison

Compares Sobel and Laplacian energy maps and displays:

- Energy maps
- Total energy values
- Comparison metrics
- Visualization chart
- Observations

### 2. 📍 Seam Detection

Displays:

- Original image
- Detected seam
- Energy map
- Seam path
- Seam length
- Total seam energy

### 3. ✂️ Seam Removal and Algorithm Comparison

Allows users to:

- Select the number of seams to remove
- Perform content-aware resizing
- Compare algorithm results
- Analyze image-processing performance

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Streamlit
- Matplotlib
- scikit-image

## 📂 Project Structure

Seam-Carving-Comparative-Algorithm-Analysis/
├── app.py
├── energy_map.py
├── seam_detection.py
├── seam_removal.py
├── requirements.txt
├── .gitignore
├── Image1.jpg
├── data.jpeg
├── newimage.jpeg
├── DAA_SEAM_CARVING_Final_Report.docx
└── README.md

## ⚙️ Installation

Clone the repository:

git clone https://github.com/vidyasri03/Seam-Carving-Comparative-Algorithm-Analysis.git

Navigate to the project directory:

cd Seam-Carving-Comparative-Algorithm-Analysis

Install the required dependencies:

pip install -r requirements.txt

## ▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

## 📷 Demo

The interactive application demonstrates:

🔗 **[Live Demo – Seam Carving Application](https://seam-carving-comparative-algorithm-analysis-sw2igsnhczjwqycvrz.streamlit.app/)**

- Energy map generation
- Sobel vs Laplacian comparison
- Seam detection
- Seam path visualization
- Seam removal
- Algorithm comparison
- Performance analysis

## 📈 Results

The application provides visual and numerical results including:

- Total energy values
- Seam length
- Seam energy
- Execution performance
- Algorithm comparison charts
- Image quality metrics

The results help identify the strengths and limitations of different Seam Carving approaches.

## 🎯 Applications

Seam Carving can be used in:

- Content-aware image resizing
- Image retargeting
- Intelligent image cropping
- Computer vision
- Digital image processing
- Multimedia applications

## 🎓 Learning Outcomes

This project helped explore:

- Image energy functions
- Seam identification
- Dynamic Programming
- Recursive algorithms
- Greedy algorithms
- Graph-based path finding
- Dijkstra's Algorithm
- OpenCV-based image processing
- Algorithm complexity analysis
- Image quality evaluation
- Interactive visualization using Streamlit

## 📚 Project Documentation

The repository contains the project report:

DAA_SEAM_CARVING_Final_Report.docx

The report includes the detailed methodology, algorithms, implementation, analysis, and results.

## 👩‍💻 Author

Vidya Sri Kammari

B.Tech – Computer Science and Engineering
2026 Graduate
