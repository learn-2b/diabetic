import os
import shutil
from sklearn.model_selection import train_test_split

# مسار الصور الأصلية
original_dataset_dir = r"C:\Users\DELL\Desktop\dataset"
classes = ['0', '1', '2', '3', '4']

# مسارات المجلدات الجديدة
base_dir = r"C:\Users\DELL\Desktop\dataset"
train_dir = os.path.join(base_dir, 'training')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# إنشاء المجلدات الجديدة
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

# توزيع الصور
for cls in classes:
    cls_dir = os.path.join(original_dataset_dir, cls)
    images = os.listdir(cls_dir)

    # تقسيم البيانات
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # نسخ الصور
    for img in train_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(test_dir, cls, img))
