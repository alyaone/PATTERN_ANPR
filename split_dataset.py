import os
import random
import shutil

# 1. Tentukan path folder utama dataset kamu (sesuai gambar yang kamu kirim)
base_dir = r"C:\Users\Alyapresilis\Documents\Kuliah Elins\SEMESTER 6\Pengenalan Pola\Project_UAS\dataset\plate_detection_dataset\plate_detection_dataset"
src_img_dir = os.path.join(base_dir, "images")
src_ann_dir = os.path.join(base_dir, "annotations")

# 2. Tentukan target folder baru sesuai standar YOLO
# Kita sekalian ubah 'annotations' menjadi 'labels' agar YOLO tidak error
target_img_train = os.path.join(base_dir, "images", "train")
target_img_val = os.path.join(base_dir, "images", "val")
target_lbl_train = os.path.join(base_dir, "labels", "train")
target_lbl_val = os.path.join(base_dir, "labels", "val")

# Buat folder-folder baru tersebut jika belum ada
for folder in [target_img_train, target_img_val, target_lbl_train, target_lbl_val]:
    os.makedirs(folder, exist_ok=True)

# 3. Ambil semua list file gambar yang ada saat ini
# Memastikan hanya membaca file, bukan membaca folder train/val yang baru dibuat
all_images = [f for f in os.listdir(src_img_dir) if os.path.isfile(os.path.join(src_img_dir, f))]

# Acak urutan file agar pembagiannya adil (random)
random.seed(42)  # Biar hasil acaknya konsisten kalau di-run ulang
random.shuffle(all_images)

# 4. Tentukan rasio split (0.8 artinya 80% untuk train, 20% untuk val)
split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)

train_images = all_images[:split_index]
val_images = all_images[split_index:]

def move_files(file_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    moved_count = 0
    for img_name in file_list:
        # Tentukan nama file label (.txt) yang cocok dengan nama gambar (.png/.jpg)
        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + ".txt"
        
        src_img = os.path.join(source_img_dir, img_name)
        src_lbl = os.path.join(source_lbl_dir, lbl_name)
        
        # Pindahkan gambar jika ada
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(dest_img_dir, img_name))
            
            # Pindahkan label pasangannya jika ada
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, os.path.join(dest_lbl_dir, lbl_name))
            
            moved_count += 1
    return moved_count

# Eksekusi pemindahan file
print("Sedang memproses pembagian dataset...")
total_train = move_files(train_images, src_img_dir, src_ann_dir, target_img_train, target_lbl_train)
total_val = move_files(val_images, src_img_dir, src_ann_dir, target_img_val, target_lbl_val)

# Hapus folder annotations lama jika sudah kosong
if os.path.exists(src_ann_dir) and not os.listdir(src_ann_dir):
    os.rmdir(src_ann_dir)
    print("Folder 'annotations' lama berhasil dibersihkan.")

print(f"Selesai! Berhasil membagi:")
print(f" - Data Training : {total_train} pasang gambar & label")
print(f" - Data Validasi : {total_val} pasang gambar & label")