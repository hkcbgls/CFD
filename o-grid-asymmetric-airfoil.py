import numpy as np
import matplotlib.pyplot as plt

def naca4_geometry(x, m, p, t, c=1.0):
    """
    m: độ cong tối đa (ví dụ 0.02)
    p: vị trí độ cong tối đa (ví dụ 0.4)
    t: độ dày tối đa (ví dụ 0.12)
    c: chiều dài dây cung (mặc định 1.0)
    """
    
    y_t = 5 * t * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) -
                       0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
    
    if p == 0:
        y_c = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        y_c = np.where(x <= p*c, 
                       (m / p**2) * (2*p*(x/c) - (x/c)**2), 
                       (m / (1-p)**2) * ((1-2*p) + 2*p*(x/c) - (x/c)**2))
        
        dyc_dx = np.where(x <= p*c, 
                          (2*m / p**2) * (p - (x/c)), 
                          (2*m / (1-p)**2) * (p - (x/c)))

    theta = np.arctan(dyc_dx)
    
    x_u = x - y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    
    x_l = x + y_t * np.sin(theta)
    y_l = y_c - y_t * np.cos(theta)
    
    return x_u, y_u, x_l, y_l

def generate_naca4_asymmetric_airfoil():
    c = 1.0            
    R = 3.0 * c
    m = 0.04
    p = 0.4
    t = 0.12       
    k=1.15
    n_chord_mesh = 30
    n_chord_geometry = 100 
    n_radial_mesh = 10  

    geom_points = np.linspace(0, c, n_chord_geometry + 1)
    x_u, y_u, x_l, y_l = naca4_geometry(geom_points, m, p, t, c)
    x_geom_points = np.concatenate([x_u[::-1], x_l[1:]])
    y_geom_points = np.concatenate([y_u[::-1], y_l[1:]])
    x_c, y_c = c/2, (y_u[n_chord_geometry//2] + y_l[n_chord_geometry//2]) / 2

    theta_mesh = np.linspace(0, np.pi, n_chord_mesh + 1)
    x_mesh_distribution = c/2 * (1 - np.cos(theta_mesh))
    xu_mesh, yu_mesh, xl_mesh, yl_mesh = naca4_geometry(x_mesh_distribution, m, p, t, c)
    x_mesh_points = np.concatenate([xu_mesh[::-1], xl_mesh[1:]])
    y_mesh_points = np.concatenate([yu_mesh[::-1], yl_mesh[1:]])

    n_total_rays = len(x_mesh_points)
    theta_farfield = np.linspace(0, 2 * np.pi, n_total_rays)
    all_rays = []
    for i in range(n_total_rays):
        x_mesh_points_1, y_mesh_points_1 = x_mesh_points[i], y_mesh_points[i]
        x_farfield = x_c + R * np.cos(theta_farfield[i])
        y_farfield = y_c + R * np.sin(theta_farfield[i])

        i_indices = np.arange(n_radial_mesh + 1)
        
        k = 1.15
        if k == 1.0:
            radial_mesh = np.linspace(0, 1, n_radial_mesh + 1)
        else:
            radial_mesh = (k**i_indices - 1.0) / (k**n_radial_mesh - 1.0)

        ray_x = x_mesh_points_1 + radial_mesh * (x_farfield - x_mesh_points_1)
        ray_y = y_mesh_points_1 + radial_mesh * (y_farfield - y_mesh_points_1)
        
        rays = np.column_stack((ray_x, ray_y))
        all_rays.append(rays)

    all_rays = np.array(all_rays)

    plt.figure(figsize=(20, 18))
    plt.plot(x_geom_points, y_geom_points, color='black', linewidth=2, label='Airfoil')
    flattened_nodes = all_rays.reshape(-1, 2)
    plt.scatter(flattened_nodes[:, 0], flattened_nodes[:, 1], color='red', s=2, label='Mesh Nodes')

    for i in range(all_rays.shape[0]):
        plt.plot(all_rays[i, :, 0], all_rays[i, :, 1], color='blue', linewidth=0.5, alpha=0.8)
        
    for j in range(all_rays.shape[1]):
        ring_x = all_rays[:, j, 0]
        ring_y = all_rays[:, j, 1]
        
        ring_x_closed = np.append(ring_x, ring_x[0])
        ring_y_closed = np.append(ring_y, ring_y[0])
        
        plt.plot(ring_x_closed, ring_y_closed, color='red', linewidth=0.5, alpha=0.8)
        
    plt.title('O-Grid Mesh: Asymmetric NACA Airfoil')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

generate_naca4_asymmetric_airfoil()