import os
import shutil
import random
RAW_DIR='model/dataset/raw/PetImages'
TRAIN_DIR='model/dataset/train'
VAL_DIR='model/dataset/val'
SPLIT=0.8
def split_dataset():
    for category in ['Dog','Cat']:
        raw_category_path=os.path.join(RAW_DIR,category)
        all_images=os.listdir(raw_category_path)
        random.shuffle(all_images)
        cutoff=int(len(all_images)*SPLIT)
        train_images=all_images[:cutoff]
        val_images=all_images[cutoff:]
        for split_name,images in[('train',train_images),('val',val_images)]:
            dest_dir=os.path.join(TRAIN_DIR if split_name=='train' else VAL_DIR,category)
            os.makedirs(dest_dir,exist_ok=True)
            for img in images:
                src=os.path.join(raw_category_path,img)
                dest=os.path.join(dest_dir,img)
                shutil.copy(src,dest)
        print(f"{category}:{len(train_images)} train,{len(val_images)} val")
split_dataset()
print("done!dataset split successfully")                