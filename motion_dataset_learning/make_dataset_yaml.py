motion_names = [
"RL_Avatar_Atk_2xCombo01_Motion",
"RL_Avatar_Atk_2xCombo02_Motion",
"RL_Avatar_Atk_2xCombo03_Motion",
"RL_Avatar_Atk_2xCombo04_Motion",
"RL_Avatar_Atk_2xCombo05_Motion",
"RL_Avatar_Atk_3xCombo01_Motion",
"RL_Avatar_Atk_3xCombo02_Motion",
"RL_Avatar_Atk_3xCombo03_Motion",
"RL_Avatar_Atk_3xCombo04_Motion",
"RL_Avatar_Atk_3xCombo05_Motion",
"RL_Avatar_Atk_3xCombo06_Motion",
"RL_Avatar_Atk_3xCombo07_Motion",
"RL_Avatar_Atk_4xCombo01_Motion",
"RL_Avatar_Atk_4xCombo03_Motion",
"RL_Avatar_Atk_Kick_Motion",
"RL_Avatar_Atk_ShieldCharge_Motion",
"RL_Avatar_Atk_ShieldSwipe01_Motion",
"RL_Avatar_Atk_ShieldSwipe02_Motion",
"RL_Avatar_Atk_SlashDown_Motion",
"RL_Avatar_Atk_SlashLeft_Motion",
"RL_Avatar_Atk_SlashRight_Motion",
"RL_Avatar_Atk_SlashUp_Motion",
"RL_Avatar_Atk_Stab_Motion",
"RL_Avatar_Counter_Atk01_Motion",
"RL_Avatar_Counter_Atk02_Motion",
"RL_Avatar_Counter_Atk03_Motion",
"RL_Avatar_Counter_Atk05_Motion",
"RL_Avatar_Dodge_Backward_Motion",
"RL_Avatar_Dodgle_Right_Motion",
"RL_Avatar_Idle_Alert_Motion",
"RL_Avatar_Idle_Battle_Motion",
"RL_Avatar_Idle_Ready_Motion",
"RL_Avatar_Kill_2xCombo01_Motion",
"RL_Avatar_Kill_2xCombo02_Motion",
"RL_Avatar_Kill_3xCombo01_Motion",
"RL_Avatar_Kill_3xCombo02_Motion",
"RL_Avatar_Kill_4xCombo01_Motion",
"RL_Avatar_RunBackward_Motion",
"RL_Avatar_RunForward_Motion",
"RL_Avatar_RunLeft_Motion",
"RL_Avatar_RunRight_Motion",
"RL_Avatar_Shield_BlockBackward_Motion",
"RL_Avatar_Shield_BlockCrouch_Motion",
"RL_Avatar_Shield_BlockDown_Motion",
"RL_Avatar_Shield_BlockLeft_Motion",
"RL_Avatar_Shield_BlockRight_Motion",
"RL_Avatar_Shield_BlockUp_Motion",
"RL_Avatar_Standoff_Circle_Motion",
"RL_Avatar_Standoff_Feint_Motion",
"RL_Avatar_Standoff_Swing_Motion",
"RL_Avatar_Sword_ParryBackward01_Motion",
"RL_Avatar_Sword_ParryBackward02_Motion",
"RL_Avatar_Sword_ParryBackward03_Motion",
"RL_Avatar_Sword_ParryBackward04_Motion",
"RL_Avatar_Sword_ParryCrouch_Motion",
"RL_Avatar_Sword_ParryDown_Motion",
"RL_Avatar_Sword_ParryLeft_Motion",
"RL_Avatar_Sword_ParryRight_Motion",
"RL_Avatar_Sword_ParryUp_Motion",
"RL_Avatar_Taunt_PoundChest_Motion",
"RL_Avatar_Taunt_Roar_Motion",
"RL_Avatar_Taunt_ShieldKnock_Motion",
"RL_Avatar_TurnLeft180_Motion",
"RL_Avatar_TurnLeft90_Motion",
"RL_Avatar_TurnRight180_Motion",
"RL_Avatar_TurnRight90_Motion",
"RL_Avatar_WalkBackward01_Motion",
"RL_Avatar_WalkBackward02_Motion",
"RL_Avatar_WalkForward01_Motion",
"RL_Avatar_WalkForward02_Motion",
"RL_Avatar_WalkLeft01_Motion",
"RL_Avatar_WalkLeft02_Motion",
"RL_Avatar_WalkRight01_Motion",
"RL_Avatar_WalkRight02_Motion",
]

motions = []

import os

recorded_motions = os.listdir("output/recordings/IQL_firstdataset")

for name in motion_names:
    if name+'.npy' in recorded_motions:
        motions.append({"file": f"\"{name}.npy\"", "weight":0.0135135})
    else:
        print(f'missing {name}.npy')

import yaml

yaml_content = {"motions": motions}

yaml.safe_dump(yaml_content, open('output/recordings/IQL_firstdataset/sword_shield_state_action.yaml','w'))