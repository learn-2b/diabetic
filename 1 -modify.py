import os
from PIL import Image

# المسار الرئيسي للمجلد
dataset_dir = r"C:\Users\DELL\Desktop\dataset"
target_size = (512, 512)  # الحجم المطلوب

# معالجة الصور في المجلدات
total_images = 0  # عداد الصور
for folder in ['training', 'validation', 'test']:
    folder_path = os.path.join(dataset_dir, folder)
    
    for cls in ['0', '1', '2', '3', '4']:
        cls_path = os.path.join(folder_path, cls)
        
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            
            try:
                total_images += 1
                print(f"جاري معالجة الصورة رقم {total_images}: {img_path}")
                
                # 1. قراءة الصورة باستخدام PIL
                img = Image.open(img_path)
                
                # 2. تغيير حجم الصورة إلى 512x512
                img_resized = img.resize(target_size)
                
                # 3. إذا كانت الصورة بصيغة PNG، تحويلها إلى JPEG
                if img_name.endswith(".png"):
                    img_resized = img_resized.convert("RGB")  # تحويل إلى صيغة RGB
                    new_img_path = img_path.replace(".png", ".jpg")
                    img_resized.save(new_img_path, "JPEG")
                    os.remove(img_path)  # حذف الصورة الأصلية بصيغة PNG
                else:
                    # حفظ الصورة بالحجم الجديد
                    img_resized.save(img_path)

                print(f"تمت معالجة الصورة بنجاح: {img_path}")
                
            except Exception as e:
                print(f"خطأ في معالجة الصورة: {img_path}, {e}")

