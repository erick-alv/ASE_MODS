HEIGHT_FOLDER_PATTERN = "[0-9][0-9][0-9]"

motion_categories = ["locomotion", "conversation", "balance", "dance", "jump"]

dataset_motions_dict = {
    "dataset_questsim": ["locomotion", "conversation", "balance"],
    "dataset_all": ["locomotion", "conversation", "balance", "dance", "jump"],
    "dataset_plusdance": ["locomotion", "conversation", "balance", "dance"],
    "dataset_plusjump": ["locomotion", "conversation", "balance", "jump"],
    "dataset_lafan": ["varied_testing"]
}

DATE_TIME_FORMAT = "_%d-%m-%H-%M-%S"

DATE_TIME_REG_PATTERN ='_[0-9][0-9](-[0-9][0-9]){4}'