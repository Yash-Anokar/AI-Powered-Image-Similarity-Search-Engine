import os
import shutil

# Your images folder
source_folder = "static\dataset"

# Count files
files = [f for f in os.listdir(source_folder) 
         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Found {len(files)} images")
print("\nHow many categories do you want?")
categories = int(input("Enter number (e.g., 3 for shirt, pants, dress): "))

print("\nEnter category names:")
category_names = []
for i in range(categories):
    name = input(f"Category {i+1} name: ")
    category_names.append(name)

# Distribute images evenly
images_per_category = len(files) // categories
print(f"\n~{images_per_category} images per category")

# Rename
idx = 0
for cat_idx, category in enumerate(category_names):
    count = images_per_category
    if cat_idx == len(category_names) - 1:  # Last category gets remainder
        count = len(files) - idx
    
    for i in range(count):
        if idx < len(files):
            old_path = os.path.join(source_folder, files[idx])
            new_name = f"{category}_{i+1:03d}{os.path.splitext(files[idx])[1]}"
            new_path = os.path.join(source_folder, new_name)
            os.rename(old_path, new_path)
            idx += 1
    
    print(f"✓ Renamed {count} images to category: {category}")

print(f"\n✅ Done! Renamed {idx} images")