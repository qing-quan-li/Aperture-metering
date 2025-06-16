import os
import tqdm
root_dir = "/home/yichengrui/data/image/new_img_TrES5/out/single/general/"
for i in range(1,2678):
    # create a directory for each star
    # os.makedirs(os.path.join(root_dir, f"star_{i}"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, f"{i}","debackgrounded"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, f"{i}","notdebackgrounded"), exist_ok=True)
    # # copy the file to the directory
    # os.system(f"cp {root_dir}/star_{i}_nobkg.fit {root_dir}/star_{i}/")
    # os.system(f"cp {root_dir}/star_{i}_with_bkg.fit {root_dir}/star_{i}/")