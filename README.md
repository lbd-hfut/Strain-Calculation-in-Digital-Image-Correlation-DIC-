# Strain-Calculation-in-Digital-Image-Correlation-DIC-
The application and comparison of deep learning methods and the subset least squares method in DIC (Digital Image Correlation) strain computation.

## Strain Calculation from Displacement Subsets

This Python function, `Strain_from_Displacement_Subset`, computes strain components (Ex, Ey, Exy) from 2D displacement fields (`u`, `v`) using a subset method. The method applies a least-squares fitting technique to calculate strain based on the local displacement gradients. The following documentation describes the function's purpose, parameters, and usage.

### Function: `Strain_f_Disp_Subset.py`

#### Purpose
The function calculates strain components (`Ex`, `Ey`, and `Exy`) from the displacement fields (`u`, `v`) using a sliding window technique. The displacement gradients within each window are estimated using least-squares fitting, and the strains are derived from these gradients.

#### Parameters
- `u`: 2D numpy array  
  The horizontal displacement field.
  
- `v`: 2D numpy array  
  The vertical displacement field.
  
- `flag`: 2D numpy array  
  A binary matrix indicating valid points in the displacement fields (`1` for valid points, `0` for invalid points). Strain calculations are only performed on valid points.
  
- `step`: float  
  The spatial resolution or step size between grid points in the displacement fields. This is used to scale the local grid when estimating displacement gradients.
  
- `SmoothLen`: int  
  The size of the window used to smooth the displacement fields. The window size should be an odd number. If an even number is provided, the function will automatically increment it by 1.

#### Returns
- `Ex`: 2D numpy array  
  The normal strain component in the x-direction.
  
- `Ey`: 2D numpy array  
  The normal strain component in the y-direction.
  
- `Exy`: 2D numpy array  
  The shear strain component.

#### Example Usage

```python
import numpy as np

# Define the displacement fields (u, v) and the flag matrix
u = np.array([[1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4]])
v = np.array([[1, 1, 1, 1],
              [3, 3, 3, 3],
              [5, 5, 5, 5],
              [7, 7, 7, 7]])
flag = np.ones((4, 4))

# Define the spatial step size and the smoothing window length
step = 1
SmoothLen = 3

# Call the function to compute strain components
Ex, Ey, Exy = Strain_from_Displacement_Subset(u, v, flag, step, SmoothLen)

# Output the results
print("Ex: \n", Ex)
print("Ey: \n", Ey)
print("Exy: \n", Exy)
```

#### Detailed Steps

1. **Input Transposition**  
   The input displacement fields (`u`, `v`) and the flag matrix are first transposed to align the data properly.

2. **Smoothing Window Adjustment**  
   The smoothing window size (`SmoothLen`) is adjusted to ensure it is an odd number, as required by the sliding window method.

3. **NaN Padding**  
   The displacement fields and the flag matrix are extended by padding the boundaries with `NaN` values. This ensures that edge pixels can still be processed.

4. **Displacement Gradients Estimation**  
   A least-squares method is used to estimate the local displacement gradients in each window.

5. **Strain Calculation**  
   The strain components are calculated from the displacement gradients.

6. **Result Cropping**  
   The extended regions of the strain matrices are trimmed to match the original displacement field size.

#### Notes
- The function assumes that the displacement fields (`u`, `v`) and the flag matrix (`flag`) are well-defined and of the same dimensions.
- The method is based on fitting a plane to the displacement data in local windows, so it requires valid data points in each window to work properly.
- If a window has fewer than 4 valid points, the function will return `NaN` for that location.
