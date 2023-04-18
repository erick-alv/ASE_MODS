import pandas as pd
import glob
import os
import shutil


def classify_cmu(xls_file_path):
    MID_CN = 'MOTION'
    DESCR_CN = 'DESCRIPTION from CMU web database'
    SUBJ_CN = 'SUBJECT from CMU web database'
    df = pd.read_excel(xls_file_path, header=10,
                       dtype={'MOTION': str,
                              'DESCRIPTION from CMU web database': str,
                              'SUBJECT from CMU web database':str}
                       )
    # eliminating nan columns
    df = df.dropna()

    #first discard by motion id
    mot_disc_cl_dict = {
        'complex_locomotion': ['105_53', '105_54', '105_55', '105_56', '105_58', '105_59', '105_60', '105_61',
                               '105_37', '105_38',
                               '91_54', '91_55', '91_56', '91_58', '91_59', '91_60', '91_61',
                               '91_37', '91_38']
    }
    disc_df_dict = {
    }
    rest_df, disc_df_dict = separate_by_actclass(df, MID_CN, mot_disc_cl_dict, disc_df_dict)

    # discard by subject; the activity that the subject does
    subj_disc_cl_dict = {
        'dance': ['dance', 'dancing', 'salsa'],
        'complex_locomotion': ['walk on uneven terrain', 'climb', 'jumps; hopscotch; sits',
                               'pushing a box; jumping off a ledge; walks', 'jumping; pushing; emotional walks',
                               'Walking with obstacles', 'Bending over', '#118 (Jumping)',
                               'Michael Jackson Styled Motions', '#120 (Various Style Walks)',
                               '#122 (Varying Height and Length Steps)', '#132 (Varying Weird Walks)',
                               '#133 (Baby Styled Walk)', '#136 (Weird Walks)',
                               '#139 (Action Walks, sneaking, wounded, looking around)', 'Stylized Walks',
                               'Action Walks, sneaking, wounded, looking around', 'Getting Up From Ground'],
        'sport': ['punch', 'kick', 'basketball', 'football', 'swim', 'sport', 'Skateboard', 'acrobatics', 'golf',
                  '#135 (Martial Arts Walks)'],
        'other': ['nursery rhymes', 'various everyday behaviors', 'stretch', 'swing', 'animal', 'construction work',
                  'suitcase', 'actor everyday activities', 'assorted motions', 'pregnant woman', 'Stylized Motions',
                  'General Subject Capture']
    }


    rest_df, disc_df_dict = separate_by_actclass(rest_df, SUBJ_CN, subj_disc_cl_dict, disc_df_dict)
    # remaining_subj = rest_df[SUBJ_CN].drop_duplicates().tolist()
    # # for el in remaining_subj:
    # #     print(el)
    # print(f"# subj: {len(remaining_subj)}")

    # then discard by description
    descr_disc_cl_dict = {
        'dance': ['dance'],
        'sport': ['basketball', 'tai', 'kick', 'punch', 'box', 'squat'],
        'complex_locomotion': ['jump', 'hop', 'roll', 'duck', 'dive', 'bend', 'sit', 'catch', 'swing',
                               'walk with anger, frustration', 'limping, hurt right leg', 'laying down', 'limp',
                               'AttitudeWalk'],
        'other': ['stepstool', 'clean', 'zombie', 'swordplay', 'story', 'mummy', 'Motorcycle', 'range of motion',
                  'pull', 'seat', 'nursery rhyme', 'unknown', 'stool', 'muscular, heavyset', 'shelters', 'passes',
                  'wrestle', 'investigating thing on ground with two hands', 'stumble', 'sexy', 'scared', 'macho',
                  'shy', 'cool', 'HurtStomach', 'ghetto', 'hurtleg', 'DragBad', 'achey', 'sad', 'amc', 'happy',
                  'excited', 'Spastic', 'Frankenstein', 'Clumsy', 'depressed', 'lavis', 'drunk', 'traffic',
                  'calibration', 'poking ground', 'searching ground']
    }
    rest_df, disc_df_dict = separate_by_actclass(rest_df, DESCR_CN, descr_disc_cl_dict, disc_df_dict)

    descr_use_cl_dict = {
        'locomotion': ['walk', 'run', 'jog',
                       '360-degree two-person whip', "blind man's bluff", 'walk forward and pick up object', 'step',
                       'march', 'Walk8','Digital8', 'walkFigure8', 'navigate around obstacles', 'turn in place'],
        'conversation': ['conversation', 'hand gestures'],
        'other': ['B comforts A', 'A comforts B', 'stumbles into', 'vignettes', 'stretching', 'careful creeping',
                  'wash self', 'Looking around', 'look around', 'ready stance', 'standing', 'wait for bus']
    }
    selected_df_dict = {}
    rest_df, selected_df_dict = separate_by_actclass(rest_df, DESCR_CN, descr_use_cl_dict, selected_df_dict)

    #remaining_descr = rest_df[DESCR_CN].drop_duplicates().tolist()
    # for el in remaining_descr:
    #     print(el)
    #print(f"# Descr: {len(remaining_descr)}")

    #add the remaining ones to the 'other' class of discarded motions
    disc_df_dict['other'] = pd.concat([disc_df_dict['other'], rest_df], ignore_index=True, sort=False)

    return selected_df_dict, disc_df_dict


def separate_by_actclass(df, CN, actclass_dict_list, acc_actclass_df_dict):
    rest_df = df
    for cl, kw_list in actclass_dict_list.items():
        for kw in kw_list:
            kw_mask = rest_df[CN].str.contains(kw, case=False, regex=False)
            kw_df = rest_df[kw_mask]
            if cl in acc_actclass_df_dict.keys():
                acc_actclass_df_dict[cl] = pd.concat([acc_actclass_df_dict[cl], kw_df], ignore_index=True, sort=False)
            else:
                acc_actclass_df_dict[cl] = kw_df

            rest_df = rest_df[~kw_mask]
    return rest_df, acc_actclass_df_dict


def copy_files_to_class_folders(folder_all_fbx, folder_store_classifications, df_dict, df_col_name, excel_name=None):
    fbx_files_list = glob.glob(glob.escape(folder_all_fbx) + "/**/*.fbx", recursive=True)

    dfs = []
    total_len = 0
    for class_name, df in df_dict.items():
        for index, row in df.iterrows():
            corresp_name = '/' + row[df_col_name] + '.fbx'
            matching_files = [f_name for f_name in fbx_files_list if f_name.endswith(corresp_name)]
            assert len(matching_files) == 1
            cp_name = os.path.join(folder_store_classifications, class_name, row[df_col_name] + '.fbx')
            dirs_cp_path = os.path.dirname(cp_name)
            if not os.path.exists(dirs_cp_path):
                os.makedirs(dirs_cp_path)
            shutil.copy(matching_files[0], cp_name)

        dfs.append(df)
    #     total_len += df.size
    # print(f"Total length: {total_len}")
    if len(dfs) > 0 and excel_name is not None:
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        final_name = os.path.join(folder_store_classifications, excel_name)
        combined_df.to_excel(final_name)




def main():
    selected_df_dict, disc_df_dict = classify_cmu(r'/home/erick/MotionProjs/cmu_bvh/cmuconvert-mb2-01-09/cmu-mocap-index-spreadsheet.xls')
    folder_all_fbx = '/home/erick/MotionProjs/cmu_fbx/'
    folder_store_classifications = '/home/erick/MotionProjs/ASE_MODS/ase/poselib/data/cmu_motions/'
    MID_CN = 'MOTION'
    copy_files_to_class_folders(folder_all_fbx, folder_store_classifications, selected_df_dict)#,
                                #MID_CN)

    #folder_store_classifications_discarded = '/home/erick/MotionProjs/ASE_MODS/ase/poselib/data/cmu_motions_discarded/'
    #copy_files_to_class_folders(folder_all_fbx, folder_store_classifications_discarded, disc_df_dict, MID_CN)




if __name__ == '__main__':
    main()


