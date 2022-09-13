VIDEO_SAVE_PATH='./dataset/'

TARGET_PATH='.'


python download_hd.py $VIDEO_SAVE_PATH ####download hd videos


python generate_frames_multiprocess.py $VIDEO_SAVE_PATH $TARGET_PATH ###extract frames to vspw data folder

