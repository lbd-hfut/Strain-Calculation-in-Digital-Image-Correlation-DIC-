import numpy as np
from scipy.linalg import cholesky, solve
import matplotlib.pyplot as plt
import scipy.io

def compute_displacement_gradients(u, v, roi, radius_strain, spacing=1):
    """
    Function to compute displacement gradients (dudx, dudy, dvdx, dvdy)
    
    Parameters:
        u, v: displacement fields (2D arrays)
        roi: Region of Interest (ROI) for which to calculate gradients
        radius_strain: Radius of strain window for computing gradients
        pixtounits: Pixel to physical units conversion factor
        spacing: Pixel spacing
        
    Returns:
        dudx, dudy, dvdx, dvdy: Computed displacement gradients
        validpoints: Array of valid points where gradient was computed
    """
    height, width = u.shape
    plot_dudx = np.zeros_like(u)
    plot_dudy = np.zeros_like(v)
    plot_dvdx = np.zeros_like(u)
    plot_dvdy = np.zeros_like(v)
    plot_validpoints = np.zeros_like(u, dtype=bool)
    
    # Iterate through each point in the region of interest (ROI)
    left,right,top,bottom = get_roi_bounds(roi)
    for x in range(left, right+1):
        for y in range(top, bottom+1):
            if roi[y, x]==0:
                continue
            # Extract subset of points (for simplicity, we use a square window around the point)
            subset_points, subset_u, subset_v = get_subset_points(u, v, roi, x, y, radius_strain)
            
            if subset_points is not None:
                # Build the system of equations for least squares fitting
                mat_LS = np.zeros((9,))
                u_vec_LS = np.zeros(3)
                v_vec_LS = np.zeros(3)
                
                for (xi, yi), ui, vi in zip(subset_points, subset_u, subset_v):
                    # Local coordinates in the window
                    x_LS = xi - x
                    y_LS = yi - y

                    # Fill the least squares matrices
                    mat_LS[0] += x_LS**2
                    mat_LS[3] += x_LS * y_LS
                    mat_LS[4] += y_LS**2
                    mat_LS[6] += x_LS
                    mat_LS[7] += y_LS

                    # Fill the displacement vectors
                    u_vec_LS[0] += x_LS * ui
                    u_vec_LS[1] += y_LS * ui
                    u_vec_LS[2] += ui

                    v_vec_LS[0] += x_LS * vi
                    v_vec_LS[1] += y_LS * vi
                    v_vec_LS[2] += vi

                # Fill symmetric parts of matrix
                mat_LS[1] = mat_LS[3]
                mat_LS[2] = mat_LS[6]
                mat_LS[5] = mat_LS[7]
                mat_LS[8] = len(subset_points)                  
                
                # Cholesky decomposition to solve the system of equations
                try:
                    L = cholesky(mat_LS[: 9].reshape((3,3)), lower=True)
                    # Solve for the gradient parameters using forward and backward substitution
                    u_grad = solve(L, u_vec_LS[:3])
                    v_grad = solve(L, v_vec_LS[:3])

                    # Store the results
                    plot_dudx[y, x] = u_grad[0] / spacing
                    plot_dudy[y, x] = u_grad[1] / spacing
                    plot_dvdx[y, x] = v_grad[0] / spacing
                    plot_dvdy[y, x] = v_grad[1] / spacing
                    plot_validpoints[y, x] = True
                except np.linalg.LinAlgError:
                    # If the matrix is not positive definite, skip this point
                    continue

    return plot_dudx, plot_dudy, plot_dvdx, plot_dvdy, plot_validpoints


def get_subset_points(u, v, roi, x, y, radius):
    """
    Extracts a subset of points around the point (x, y) within a given radius
    """
    height, width = u.shape
    points = []
    u_vals = []
    v_vals = []
    
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            xi, yi = x + dx, y + dy
            if 0 <= xi < width and 0 <= yi < height:
                if roi[yi, xi] or ~np.isnan(u):
                    points.append((xi, yi))
                    u_vals.append(u[yi, xi])
                    v_vals.append(v[yi, xi])
    
    if len(points) > 8:
        return points, u_vals, v_vals
    else:
        return None, None, None

def get_roi_bounds(roi_matrix):
    """
    Calculate the bounding box of an ROI in a boolean matrix.
    Args:
        roi_matrix (numpy.ndarray): A 2D boolean array where `True` represents the ROI region.
        
    Returns:
        tuple: (min_row, max_row, min_col, max_col) representing the bounding box of the ROI.
    """
    if not isinstance(roi_matrix, np.ndarray) or roi_matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Get indices of non-False (True) values
    rows, cols = np.where(roi_matrix)
    
    if len(rows) == 0 or len(cols) == 0:
        # No True values found, return None or an invalid bounding box
        return None

    # Calculate bounds
    min_row = rows.min()
    max_row = rows.max()
    min_col = cols.min()
    max_col = cols.max()
    
    return min_col, max_col, min_row, max_row

if __name__ == "__main__":
    # Simulate some displacement fields (u and v) for testing
    height, width = 256, 1024
    # u = np.random.randn(height, width)  # Horizontal displacement field
    # v = np.random.randn(height, width)  # Vertical displacement field
    mat_data = scipy.io.loadmat('./test_data/uvmat/fpb_displacement.mat')
    u = mat_data['u']
    v = mat_data['v']
    
    # Define a region of interest (ROI) with a bounding box
    roi = np.ones_like(u, dtype=bool)#; roi[20:256-20,20:1024-20]=True
    
    # Define strain window radius and conversion factors
    radius_strain = 1
    pixtounits = 1.0  # No conversion for simplicity
    spacing = 1  # 1 pixel spacing
    
    # Compute the displacement gradients
    dudx, dudy, dvdx, dvdy, validpoints = compute_displacement_gradients(u, v, roi, radius_strain)
    
    # Visualize the results
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(dudx, cmap='jet')
    ax[0].set_title('dudx')
    ax[1].imshow(dudy, cmap='jet')
    ax[1].set_title('dudy')
    ax[2].imshow(dvdx, cmap='jet')
    ax[2].set_title('dvdx')
    ax[3].imshow(dvdy, cmap='jet')
    ax[3].set_title('dvdy')
    plt.show()