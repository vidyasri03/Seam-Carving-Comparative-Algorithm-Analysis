import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time
from energy_map import compute_energy_map_sobel, compute_energy_map_laplacian, apply_red_colormap
from seam_detection import detect_seams
from seam_removal import seam_carving_dynamic, seam_carving_graph, seam_carving_greedy, seam_carving_recursive

ALGORITHMS = {
    "Greedy": seam_carving_greedy,
    "Recursive": seam_carving_recursive,
    "Dynamic Programming": seam_carving_dynamic,
    "Graph-Based (Dijkstra)": seam_carving_graph,
}





def mark_seam_on_image(image, seam_coords):
    seam_image = image.copy()  # Create a copy of the image to mark the seam
    for row, col in seam_coords:
        # Mark the seam with red color (BGR format)
        seam_image[row, col] = [0, 0, 255]
    return seam_image


# Function to display the comparison metrics
def display_comparison_metrics(sobel_energy_map, laplacian_energy_map):
    # Calculate the sum of energy values
    sobel_sum = np.sum(sobel_energy_map)
    laplacian_sum = np.sum(laplacian_energy_map)

    # Prepare data for the table
    metrics = {
        "Metric": ["Sum of Energy Values"],
        "Sobel": [sobel_sum],
        "Laplacian": [laplacian_sum],
        "Better Method": [
            "Sobel ✅" if sobel_sum > laplacian_sum else "Laplacian ✅",
        ],
    }
    df_metrics = pd.DataFrame(metrics)

    # Display as a table
    st.write("### Comparison Metrics (Sum of Energy Values)")
    st.table(df_metrics)

    # Bar chart for visual comparison
    st.write("### Metric Comparison Chart - Sum of Energy Values")
    chart_data = pd.DataFrame(
        {
            "Method": ["Sobel", "Laplacian"],
            "Sum of Energy": [sobel_sum, laplacian_sum],
        }
    )
    chart_data.set_index("Method", inplace=True)
    st.bar_chart(chart_data)

    # Expandable section for insights
    with st.expander("Insights"):
        st.write("### Observations")
        if sobel_sum > laplacian_sum:
            st.markdown("- **Sobel** detects higher total energy intensity, indicating more edges captured.")
        else:
            st.markdown("- **Laplacian** detects higher total energy intensity, indicating more edges captured.")


# Function for energy map calculation and comparison
def energy_map_page():
    st.title("Seam Carving - Energy Map Comparison")
    
    # Upload an image
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Load the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Save the image and Sobel energy map to session state
        sobel_energy_map = compute_energy_map_sobel(image)
        st.session_state.image = image
        st.session_state.sobel_energy_map = sobel_energy_map
        
        # Calculate energy maps
        laplacian_energy_map = compute_energy_map_laplacian(image)
        
        # Apply red colormap to energy maps
        sobel_red_map = apply_red_colormap(sobel_energy_map)
        laplacian_red_map = apply_red_colormap(laplacian_energy_map)
        
        # Display energy maps side by side
        st.write("### Energy Map Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(sobel_red_map, caption="Sobel Energy Map (Red)", use_container_width=True)
        with col2:
            st.image(laplacian_red_map, caption="Laplacian Energy Map (Red)", use_container_width=True)
        
        # Call function to display comparison metrics (focused on sum)
        display_comparison_metrics(sobel_energy_map, laplacian_energy_map)

        # Save energy maps to session state for later use
        st.session_state.laplacian_energy_map = laplacian_energy_map
        st.session_state.sobel_red_map = sobel_red_map
        st.session_state.laplacian_red_map = laplacian_red_map


# Function for seam detection with additional features (side-by-side comparison)
def seam_detection_page():
    st.title("Seam Detection")

    if 'sobel_energy_map' not in st.session_state:
        st.error("Please upload an image first on the Energy Map Calculation page.")
        return
    
    # Get the Sobel energy map from session state
    sobel_energy_map = st.session_state.sobel_energy_map
    image = st.session_state.image
    
    # Detect seams using Sobel energy map
    seam_coords = detect_seams(sobel_energy_map)
    
    # Mark the seam on the image
    seam_image = mark_seam_on_image(image, seam_coords)

    # Display side-by-side comparison of original image and image with detected seam
    st.write("### Original Image vs Image with Detected Seam")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(seam_image, caption="Image with Detected Seam", use_container_width=True)
    
    # Display energy map visualization alongside the seam
    st.write("### Energy Map and Seam Path")
    col3, col4 = st.columns(2)
    
    with col3:
        st.image(st.session_state.sobel_red_map, caption="Sobel Energy Map (Red)", use_container_width=True)
    with col4:
        st.image(seam_image, caption="Image with Seam", use_container_width=True)
    
    # Display seam length and energy
    seam_length = len(seam_coords)
    seam_energy = sum(sobel_energy_map[row, col] for row, col in seam_coords)
    st.write(f"### Seam Information")
    st.write(f"- **Seam Length**: {seam_length} pixels")
    st.write(f"- **Total Seam Energy**: {seam_energy}")




def calculate_psnr(original, processed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        original (np.array): Original image.
        processed (np.array): Processed image.
    
    Returns:
        float: PSNR value in decibels (dB).
    """
    # Resize processed image to match original's dimensions
    processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    mse = np.mean((original - processed_resized) ** 2)
    if mse == 0:  # If MSE is zero, return a very high PSNR
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_ssim(original, processed):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Args:
        original (np.array): Original image.
        processed (np.array): Processed image.
    
    Returns:
        float: SSIM value.
    """
    # Resize processed image to match original's dimensions
    processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for SSIM calculation
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed_resized, cv2.COLOR_BGR2GRAY)
    
    ssim_value = ssim(original_gray, processed_gray)
    return ssim_value

# Updated comparison function to include runtime, PSNR, and SSIM
def compare_algorithms_with_metrics(image, num_seams):
    results = {}

    for algo_name, algo_func in ALGORITHMS.items():
        # Measure runtime
        start_time = time.time()
        processed_image = algo_func(image.copy(), num_seams)
        runtime = time.time() - start_time

        # Calculate PSNR
        psnr_value = calculate_psnr(image, processed_image)
        
        # Calculate SSIM
        ssim_value = calculate_ssim(image, processed_image)
        # Save results
        results[algo_name] = {
            "Processed Image": processed_image,
            "Runtime": runtime,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
        }

    return results

# Updated page to include PSNR, SSIM, and runtime in the comparison
# Updated Seam Removal Page with Algorithm Comparison and Best Algorithm Highlight
def seam_removal_page():
    if 'image' not in st.session_state:
        st.error("Please upload an image first!")
        return
    
    st.title("Seam Removal and Algorithm Comparison")

    # Retrieve the image from session state
    image = st.session_state.image
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_container_width=True)
    
    # Number of seams to remove
    num_seams = st.slider("Number of Seams to Remove", 1, 100, 10)

    if st.button("Run Seam Removal"):
        # Compare algorithms
        results = compare_algorithms_with_metrics(image, num_seams)

        # Display results
        st.write("### Seam Removal Results")

        # Display images side by side
        st.write("#### Processed Images by Algorithm")
        col1, col2 = st.columns(2)
        for i, (algo_name, result) in enumerate(results.items()):
            if i % 2 == 0:
                with col1:
                    st.image(result["Processed Image"], caption=f"{algo_name}", use_container_width=True)
            else:
                with col2:
                    st.image(result["Processed Image"], caption=f"{algo_name}", use_container_width=True)

        # Create a comparison table for metrics
        st.write("### Comparison Table")
        metrics_data = {
            "Algorithm": [],
            "Runtime (seconds)": [],
            "PSNR (dB)": [],
            "SSIM": [],
        }

        for algo_name, result in results.items():
            metrics_data["Algorithm"].append(algo_name)
            metrics_data["Runtime (seconds)"].append(result['Runtime'])
            metrics_data["PSNR (dB)"].append(result['PSNR'])
            metrics_data["SSIM"].append(result['SSIM'])

        df_metrics = pd.DataFrame(metrics_data)

        # Display the comparison table
        st.table(df_metrics)

        # Add bar charts for comparison
        st.write("### Runtime Comparison (Bar Chart)")
        runtime_chart_data = df_metrics[["Algorithm", "Runtime (seconds)"]].copy()
        runtime_chart_data.set_index("Algorithm", inplace=True)
        st.bar_chart(runtime_chart_data)

        st.write("### PSNR Comparison (Bar Chart)")
        psnr_chart_data = df_metrics[["Algorithm", "PSNR (dB)"]].copy()
        psnr_chart_data.set_index("Algorithm", inplace=True)
        st.bar_chart(psnr_chart_data)

        st.write("### SSIM Comparison (Bar Chart)")
        ssim_chart_data = df_metrics[["Algorithm", "SSIM"]].copy()
        ssim_chart_data.set_index("Algorithm", inplace=True)
        st.bar_chart(ssim_chart_data)

        # Determine the best algorithm
        st.write("### Best Algorithm Based on Metrics")
        df_metrics['Rank (Runtime)'] = df_metrics['Runtime (seconds)'].rank(ascending=True)
        df_metrics['Rank (PSNR)'] = df_metrics['PSNR (dB)'].rank(ascending=False)
        df_metrics['Rank (SSIM)'] = df_metrics['SSIM'].rank(ascending=False)

        # Calculate total score (lower is better)
        df_metrics['Total Rank'] = df_metrics[['Rank (Runtime)', 'Rank (PSNR)', 'Rank (SSIM)']].sum(axis=1)
        best_algorithm = df_metrics.loc[df_metrics['Total Rank'].idxmin()]

        # Display best algorithm details
        st.success(f"The best algorithm is **{best_algorithm['Algorithm']}** "
                   f"with total rank score of {best_algorithm['Total Rank']:.2f}.")
        st.write("#### Best Algorithm Metrics")
        st.table(best_algorithm[['Runtime (seconds)', 'PSNR (dB)', 'SSIM']])

def main():
    st.sidebar.title("Seam Carving and Image Processing")
    page = st.sidebar.radio("Select a Page", ("Energy Map Comparison", "Seam Detection", "Seam Removal"))

    if page == "Energy Map Comparison":
        energy_map_page()
    elif page == "Seam Detection":
        seam_detection_page()
    elif page == "Seam Removal":
        seam_removal_page()

if __name__ == "__main__":
    main()
