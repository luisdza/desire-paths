import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Streamlit app setup
st.title("Desire Path Detection Using A* Search")
st.write("Upload a satellite image and watch the A* algorithm detect a desire path.")

# A* Algorithm Implementation
def a_star_search(start, end, grid):
    rows, cols = grid.shape
    open_set = PriorityQueue()  # Priority queue to keep track of nodes to explore
    open_set.put((0, start))
    came_from = {}  # Dictionary to store the path
    g_score = {start: 0}  # Cost from start to the current node
    f_score = {start: heuristic(start, end)}  # Estimated cost from start to end through the current node

    while not open_set.empty():
        current = open_set.get()[1]  # Get the node with the lowest f_score

        if current == end:
            return reconstruct_path(came_from, current)  # Reconstruct the path if the end is reached

        neighbors = get_neighbors(current, rows, cols)  # Get all valid neighbors of the current node
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + cost(neighbor, grid)  # Calculate tentative g_score
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current  # Update path
                g_score[neighbor] = tentative_g_score  # Update g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)  # Update f_score
                open_set.put((f_score[neighbor], neighbor))  # Add neighbor to the open set
    return []  # Return an empty path if there's no valid path

# Heuristic function to estimate the cost from a to b (Manhattan distance)
def heuristic(a, b):
    h = abs(a[0] - b[0]) + abs(a[1] - b[1])
    return h

# Cost function to determine the movement cost based on pixel intensity
def cost(node, grid):
    c = 1.0 if grid[node[0], node[1]] > 128 else 10.0  # Lower cost for brighter pixels, higher for darker pixels
    return c

# Function to get valid neighbors of the current position
def get_neighbors(pos, rows, cols):
    x, y = pos
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]  # Potential neighbors (up, down, left, right)
    valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < rows and 0 <= ny < cols]  # Filter out-of-bounds neighbors
    return valid_neighbors

# Function to reconstruct the path from the start to the end node
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()  # Reverse the path to get it from start to end
    return path

# File uploader for the satellite image
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image from the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)  # Decode the uploaded image
    image = original_image.copy()  # Make a copy of the original image for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("Applied Gaussian Blur to the image.")

    # Allow the user to adjust Canny edge detection thresholds
    threshold1 = st.slider("Canny Edge Detection - Threshold 1", min_value=0, max_value=255, value=50)
    threshold2 = st.slider("Canny Edge Detection - Threshold 2", min_value=0, max_value=255, value=150)
    print(f"Canny thresholds set to: threshold1={threshold1}, threshold2={threshold2}")

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=threshold1, threshold2=threshold2)
    print("Applied Canny edge detection.")

    # Set start and end points for the pathfinding algorithm
    start = (10, 10)  # Top-left corner
    end = (edges.shape[0] - 10, edges.shape[1] - 10)  # Bottom-right corner
    print(f"Start point: {start}, End point: {end}")

    # Find the path using A* search
    path = a_star_search(start, end, edges)

    # Highlight the detected path in red on the processed image
    if path:
        path_coords = np.array(path)
        image[path_coords[:, 0], path_coords[:, 1]] = [0, 0, 255]  # Mark the path in red (BGR format)
        print("Path highlighted on the image.")
    else:
        print("No path detected.")

    # Display the original and processed images side by side using Streamlit
    st.subheader("Original Image")
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    st.subheader("Detected Desire Path (Highlighted in Red)")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
