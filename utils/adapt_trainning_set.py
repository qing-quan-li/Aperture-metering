import pandas as pd
import glob
origin_dataset_path = "D:\\project\\new_img_TrES5\\out\\catalog\\merged_star_catalog.csv"
ready_dataset_path = "D:\\project\\new_img_TrES5\\out\\catalog\\ready_star_catalog.csv"
origin_data = pd.read_csv(origin_dataset_path)
input_star_dir = "D:\\project\\new_img_TrES5\\out\\stacked\\rotated_notdebackgrounded\\"
rename_dict = {"best_good_inner_radius":"aperture_radius","best_good_middle_radius":"inner_radius","best_good_outer_radius":"outer_radius"}
# rename the columns of origin_data
for old_name, new_name in rename_dict.items():
    if old_name in origin_data.columns:
        origin_data.rename(columns={old_name: new_name}, inplace=True)

print(origin_data.columns)
new_row_list = []
# Iterate through each row in the DataFrame
for index, row in origin_data.iterrows():
    print(f"Processing row {index+1}/{len(origin_data)}")
    star_id_this = int(row['star_id'])
    path_this_list = glob.glob(input_star_dir + f"rot_*_star_{star_id_this}_*")
    #print(path_this_list)
    for path_this in path_this_list:

        # Create a new row with the extracted star_id and other data from the original row
        new_row = {
            'star_id': star_id_this,
            'aperture_radius': row['aperture_radius'],
            'inner_radius': row['inner_radius'],
            'outer_radius': row['outer_radius'],
            'cutout_filename': path_this,
            'feasible': int(row['feasible'])
        }
        new_row_list.append(new_row)
    # if index>100:
    #     break
#print(new_row_list)
# Create a new DataFrame from the list of new rows
new_data = pd.DataFrame(new_row_list)
# Save the new DataFrame to a CSV file
print(new_data.iloc[0]['cutout_filename'])
new_data.to_csv(ready_dataset_path, index=False)
