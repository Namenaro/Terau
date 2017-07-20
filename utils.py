# -*- coding: utf-8 -*
import os
import glob

def setup_folder_for_results(main_folder='results'):
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    else:
        files = glob.glob('/' + main_folder + '/*')
        for f in files:
            os.remove(f)
    os.chdir(main_folder)