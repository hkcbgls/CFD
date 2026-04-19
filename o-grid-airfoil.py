import numpy as np
import matplotlib.pyplot as plt

def naca4_symmetric(x, c=1.0, t=0.12):
    """Tính toán tọa độ y cho airfoil NACA 4 số đối xứng"""
    # ... (Giữ nguyên phương trình như cũ)
    y = 5 * t * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 
                     0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
    return y

def generate_and_visualize_mesh_steps():
    # --- THÔNG SỐ ĐẦU VÀO ---
    c = 1.0                  # Chiều dài dây cung
    R = 3.0 * c               # Giảm bán kính biên ngoài xuống một chút để dễ nhìn điểm
    n_chord_mesh = 30        # Số đoạn lưới trên dây cung
    n_radial_mesh = 10       # Số đoạn lưới theo hướng bán kính (n)
    
    xc, yc = c / 2.0, 0.0    
    
    # 1. ĐỊNH NGHĨA HÌNH HỌC (Để vẽ)
    # ... (Giữ nguyên phần vẽ Airfoil và Far-field mịn như cũ)
    beta_geom = np.linspace(0, np.pi, 200)
    x_geom = (c / 2.0) * (1.0 - np.cos(beta_geom))
    y_geom_full = np.concatenate([naca4_symmetric(x_geom, c)[::-1], -naca4_symmetric(x_geom, c)[1:]])
    x_geom_full = np.concatenate([x_geom[::-1], x_geom[1:]])
    
    # 2. SINH LƯỚI (MESH GENERATION)
    # 2.1 Nodes trên bề mặt (Cosine)
    beta_mesh = np.linspace(0, np.pi, n_chord_mesh + 1)
    x_mesh = (c / 2.0) * (1.0 - np.cos(beta_mesh))
    x_mesh_full = np.concatenate([x_mesh[::-1], x_mesh[1:]])
    y_mesh_full = np.concatenate([naca4_symmetric(x_mesh, c)[::-1], -naca4_symmetric(x_mesh, c)[1:]])
    
    n_total_rays = len(x_mesh_full)
    theta_farfield = np.linspace(0, 2 * np.pi, n_total_rays)
    
    # mesh_points sẽ lưu tọa độ của TẤT CẢ các điểm nút.
    # Kích thước: (số_tia, n+1, 2)
    all_mesh_nodes = []
    
    for i in range(n_total_rays):
        # Điểm P trên airfoil, Điểm F trên far-field
        px, py = x_mesh_full[i], y_mesh_full[i]
        fx = xc + R * np.cos(theta_farfield[i])
        fy = yc + R * np.sin(theta_farfield[i])
        
        # TẠO CÁC ĐIỂM CHIA TRÊN TIA (Dùng linspace - chia đều)
        # s chạy từ 0 đến 1, gồm n+1 điểm
        s = np.linspace(0, 1, n_radial_mesh + 1)
        
        # Công thức nội suy vector để tìm tọa độ (x, y) của n+1 điểm trên tia thứ i
        ray_nodes_x = px + s * (fx - px)
        ray_nodes_y = py + s * (fy - py)
        
        # Gom lại thành ma trận (n+1, 2) cho tia này
        ray_nodes = np.column_stack((ray_nodes_x, ray_nodes_y))
        all_mesh_nodes.append(ray_nodes)
        
    all_mesh_nodes = np.array(all_mesh_nodes)
    
    # --- TRỰC QUAN HÓA (VISUALIZATION) ---
    plt.figure(figsize=(12, 10))
    
    # A. Vẽ biên dạng Airfoil (Màu đen đậm)
    plt.plot(x_geom_full, y_geom_full, color='black', linewidth=2.5, label='Airfoil')
    
    # B. [BƯỚC 1 CỦA BẠN]: VẼ TẤT CẢ CÁC ĐIỂM NÚT LƯỚI (MESH NODES)
    # Dùng plt.scatter để vẽ dấu chấm. 
    # 's=5' là kích thước điểm, 'alpha=0.5' để làm mờ chút cho dễ nhìn xuyên thấu.
    # Ta cần duỗi phẳng mảng nodes ra (reshape) để vẽScatter cho nhanh.
    flattened_nodes = all_mesh_nodes.reshape(-1, 2)
    plt.scatter(flattened_nodes[:, 0], flattened_nodes[:, 1], color='blue', s=5, alpha=0.5, label='Mesh Nodes')
    
    # C. Vẽ các đường tia (Radial Lines) - (Màu xanh lam mờ)
    for i in range(all_mesh_nodes.shape[0]):
        plt.plot(all_mesh_nodes[i, :, 0], all_mesh_nodes[i, :, 1], color='blue', linewidth=0.3, alpha=0.3)
        
    # D. [BƯỚC 2 CỦA BẠN]: NỐI ĐIỂM TẠO ĐƯỜNG BAO (O-RINGS)
    # Ta duyệt qua từng 'vòng' (chỉ số j chạy từ 0 đến n).
    # Với mỗi vòng j, ta nối điểm thứ j của TẤT CẢ các tia lại với nhau.
    for j in range(all_mesh_nodes.shape[1]):
        # Bóc tách tọa độ X và Y của vòng thứ j
        # Tuyến đường: Tia 0 -> Tia 1 -> ... -> Tia N
        ring_x = all_mesh_nodes[:, j, 0]
        ring_y = all_mesh_nodes[:, j, 1]
        
        # ĐỂ VÒNG LƯỚI KHÉP KÍN (O-Grid):
        # Ta phải nối điểm cuối cùng (của tia cuối) về lại điểm đầu tiên (của tia 0).
        ring_x_closed = np.append(ring_x, ring_x[0])
        ring_y_closed = np.append(ring_y, ring_y[0])
        
        # Vẽ đường bao màu đỏ
        plt.plot(ring_x_closed, ring_y_closed, color='red', linewidth=0.5, alpha=0.8)
        
    plt.title('Visions of Mesh: Showing Nodes and Connecting O-Rings')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

generate_and_visualize_mesh_steps()