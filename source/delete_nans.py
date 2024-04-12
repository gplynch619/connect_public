import os
import sys
import numpy as np

root_dir = sys.argv[1]

directories = []

for fname in os.listdir(root_dir):
    d = os.path.join(root_dir, fname)
    if os.path.isdir(d):
        directories.append(d)

n_ranks = np.arange(1,19)
lines_to_delete = {d:{} for d in directories}

for directory in directories:
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            if "nan" in open(os.path.join(directory, fname), "r").read():
                lines_to_delete[directory][fname] = []
                with open(os.path.join(directory, fname), "r") as f: 
                    line_number = 0
                    for line in f:
                        if "nan" in line:
                            lines_to_delete[directory][fname].append(line_number)
                        line_number+=1

#for directory in lines_to_delete.keys():
#    for fname in lines_to_delete[directory].keys():
        #print("{}: {}".format(fname, lines_to_delete[directory][fname]))

dict_by_number = {i:{} for i in np.arange(1,19)}
for directory in lines_to_delete.keys():
    for fname in lines_to_delete[directory].keys():
        number = fname.split("_")[-1].split(".")[0]
        data_type = fname.split("_")[1]
        dict_by_number[int(number)][data_type] = lines_to_delete[directory][fname]

total_ind_by_number =  {i:{} for i in np.arange(1,19)}
for k,v in dict_by_number.items():
    total = set()
    for var, ind_list in v.items():
        total = total.union(set(ind_list))
    total_ind_by_number[k] = np.sort(list(total))

print(total_ind_by_number)

for directory in directories:
    dir_name = directory.split("/")[-1]
    if ("model_params_data" in dir_name) or ("derived_data" in dir_name):
        for fname in os.listdir(directory):
            number = int(fname.split("_")[-1].split(".")[0])
            if number in total_ind_by_number.keys():
                f2name = fname.split(".")[0]+"_new.txt"
                with open(os.path.join(directory, fname), "r") as f1:
                    lines = f1.readlines()
                with open(os.path.join(directory, fname), "w") as f2:
                    line_count = 0
                    for line in lines:
                        if ((2*line_count-1) in total_ind_by_number[number]):
                            print("pass")
                        else:
                            f2.write(line)
                        line_count+=1
    else:
        for fname in os.listdir(directory):
            number = int(fname.split("_")[-1].split(".")[0])
            if number in total_ind_by_number.keys():
                f2name = fname.split(".")[0]+"_new.txt"
                with open(os.path.join(directory, fname), "r") as f1:
                    lines = f1.readlines()
                with open(os.path.join(directory, fname), "w") as f2:
                    line_count = 0
                    for line in lines:
                        if ((line_count+1) in total_ind_by_number[number]) or (line_count in total_ind_by_number[number]):
                            print("pass")
                        else:
                            f2.write(line)
                        line_count+=1
                
#    if (v["tt"]==v["te"]) and  (v["te"]==v["ee"]) and  (v["ee"]==v["pp"]):
#        print("True for {}".format(k))
#print(set(dir_with_nan))
#with open(target_dir, "r") as f:
#    i=0
#    for line in f:
#        if "nan" in line:
#            print(i) 
#        i+=1   
