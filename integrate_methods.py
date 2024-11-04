import torch

def trapezoidal_rule(f_values, h):
    """
    使用梯形法计算一维积分
    f_values: torch.Tensor, 表示函数值（例如 Z_x 或 Z_y 的一行或一列）
    h: 网格步长
    """
    return h * (0.5 * f_values[0] + torch.sum(f_values[1:-1]) + 0.5 * f_values[-1])

def romberg_integration_1d(f_values, h, tol=1e-6, max_order=10):
    """
    龙贝格积分法计算一维积分
    f_values: torch.Tensor, 表示函数值（例如 Z_x 或 Z_y 的一行或一列）
    h: 网格步长
    tol: 误差容限
    max_order: 最大阶数
    """
    R = torch.zeros((max_order, max_order))
    R[0, 0] = trapezoidal_rule(f_values, h)
    
    for k in range(1, max_order):
        n = 2**k
        R[k, 0] = trapezoidal_rule(f_values, h / n)
        
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
        
        if torch.abs(R[k, k] - R[k - 1, k - 1]) < tol:
            return R[k, k]
    
    return R[max_order - 1, max_order - 1]

def romberg_integration_2d(Z_x, Z_y, dx, dy, tol=1e-6):
    """
    根据 Z_x 和 Z_y 计算矩阵 Z
    Z_x: torch.Tensor, 表示沿 x 方向的偏导数矩阵
    Z_y: torch.Tensor, 表示沿 y 方向的偏导数矩阵
    dx, dy: x 和 y 方向的步长
    tol: 误差容限
    """
    m, n = Z_x.shape
    
    # 初始化 Z 矩阵
    Z = torch.zeros_like(Z_x)
    
    # 沿 x 方向积分（逐行计算）
    for i in range(m):
        Z[i, :] = romberg_integration_1d(Z_x[i, :], dx, tol)
    
    # 沿 y 方向积分（逐列计算）
    for j in range(n):
        Z[:, j] += romberg_integration_1d(Z_y[:, j], dy, tol)
    
    return Z

def trapezoidal_integrate_2d(Z_x, Z_y, dx, dy):
    """
    使用梯形法从偏导数场 Z_x 和 Z_y 恢复矩阵 Z
    Z_x: torch.Tensor, 沿 x 方向的偏导数场
    Z_y: torch.Tensor, 沿 y 方向的偏导数场
    dx, dy: x 和 y 方向的步长
    """
    m, n = Z_x.shape
    
    # 初始化 Z 矩阵
    Z = torch.zeros_like(Z_x)
    
    # 沿 x 方向积分（逐行计算）
    for i in range(m):
        Z[i, 0] = 0  # 假设初始值为 0
        for j in range(1, n):
            Z[i, j] = Z[i, j - 1] + 0.5 * (Z_x[i, j] + Z_x[i, j - 1]) * dx
    
    # 沿 y 方向积分（逐列计算）
    for j in range(n):
        for i in range(1, m):
            Z[i, j] += 0.5 * (Z_y[i, j] + Z_y[i - 1, j]) * dy
    
    return Z

def simpson_integrate_2d(Z_x, Z_y, dx, dy):
    """
    使用辛普森法从偏导数场 Z_x 和 Z_y 恢复矩阵 Z
    Z_x: torch.Tensor, 沿 x 方向的偏导数场
    Z_y: torch.Tensor, 沿 y 方向的偏导数场
    dx, dy: x 和 y 方向的步长
    """
    m, n = Z_x.shape
    
    # 初始化 Z 矩阵
    Z = torch.zeros_like(Z_x)
    
    # 沿 x 方向积分（逐行计算）
    for i in range(m):
        Z[i, 0] = 0  # 假设初始值为 0
        for j in range(1, n, 2):
            if j + 1 < n:
                Z[i, j+1] = Z[i, j - 1] + (dx / 3) * (Z_x[i, j - 1] + 4 * Z_x[i, j] + Z_x[i, j + 1])
    
    # 沿 y 方向积分（逐列计算）
    for j in range(n):
        for i in range(1, m, 2):
            if i + 1 < m:
                Z[i+1, j] += (dy / 3) * (Z_y[i-1, j] + 4 * Z_y[i, j] + Z_y[i+1, j])
    
    return Z

def monte_carlo_integrate_2d(Z_x, Z_y, dx, dy, num_samples=10000):
    """
    使用蒙特卡罗法从偏导数场 Z_x 和 Z_y 恢复矩阵 Z
    Z_x: torch.Tensor, 沿 x 方向的偏导数场
    Z_y: torch.Tensor, 沿 y 方向的偏导数场
    dx, dy: x 和 y 方向的步长
    num_samples: 采样点数量
    """
    m, n = Z_x.shape
    Z = torch.zeros_like(Z_x)
    
    # 随机采样积分区域中的点
    x_samples = torch.randint(0, n, (num_samples,))
    y_samples = torch.randint(0, m, (num_samples,))
    
    # 计算 Z_x 和 Z_y 在这些采样点上的平均值
    Z_x_avg = Z_x[y_samples, x_samples].mean()
    Z_y_avg = Z_y[y_samples, x_samples].mean()
    
    # 恢复 Z 矩阵
    Z = Z_x_avg * dx + Z_y_avg * dy
    
    return Z
