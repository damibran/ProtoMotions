import yaml
import os

root_dir = "output/recordings/IQL_seconddataset"

recorded_motions = os.listdir(root_dir)

"""
for name in recorded_motions:
    ind = name.find("_actions")
    if ind != -1:
        new_name = name[:ind] + name[ind + len("_actions"):name.find(".")] + "_actions.npy"
        os.rename(root_dir + '/' + name, root_dir + '/' + new_name)
    print(name)
"""
for i in range(4):
    motions = []

    for name in recorded_motions:
        if name.find(f"_{i}.npy") != -1:
            motions.append({"file": f"{name}", "weight": 0.0135135})

    yaml_content = {"motions": motions}

    yaml.safe_dump(yaml_content, open(f'output/recordings/IQL_seconddataset/sword_shield_state_action_{i}.yaml','w'))