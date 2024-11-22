import os
import shutil
import random

def split_data_to_valid(train_dir, valid_dir, valid_ratio=0.2):
    # Đường dẫn thư mục images và labels
    images_dir = os.path.join(train_dir, 'images')
    labels_dir = os.path.join(train_dir, 'labels')
    valid_images_dir = os.path.join(valid_dir, 'images')
    valid_labels_dir = os.path.join(valid_dir, 'labels')
    
    # Tạo thư mục valid nếu chưa tồn tại
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    
    # Lấy danh sách tất cả file ảnh
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    valid_size = int(len(image_files) * valid_ratio)
    
    # Chọn ngẫu nhiên các file để đưa vào tập valid
    valid_files = random.sample(image_files, valid_size)
    
    # Di chuyển file vào thư mục valid
    for file_name in valid_files:
        image_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        valid_image_path = os.path.join(valid_images_dir, file_name)
        valid_label_path = os.path.join(valid_labels_dir, os.path.basename(label_path))
        
        if os.path.exists(image_path):
            shutil.move(image_path, valid_image_path)
        if os.path.exists(label_path):
            shutil.move(label_path, valid_label_path)

    print("Chia tập dữ liệu thành công!")

# Sử dụng hàm
train_dir = 'D:/Study/DACN/data/data_img/train'
valid_dir = 'D:/Study/DACN/data/data_img/valid'
split_data_to_valid(train_dir, valid_dir, valid_ratio=0.2)
