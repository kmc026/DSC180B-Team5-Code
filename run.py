import sys
import json

from src.main import write_ssim
from src.image_processing_pipeline import *


def main(targets):

    test_config = json.load(open("config/config.json"))

    pipeline_config = json.load(open("config/Image-Processing-Pipeline-config.json"))

    if 'test' in targets:
        
        write_ssim(test_config['actual_image_dir'], test_config['ground_truth_image_dir'], test_config['SSIM_result_dir'])

        image_processing_and_print("config/Image-Processing-Pipeline-config.json", pipeline_config['overall_result_dir'], pipeline_config['canny_result_dir'])

    if 'ssim-text' in targets:

        write_ssim(test_config['actual_image_dir'], test_config['ground_truth_image_dir'], test_config['SSIM_result_dir'])

    if 'ssim-graph' in targets:

        image_processing_and_print("config/Image-Processing-Pipeline-config.json", pipeline_config['overall_result_dir'], pipeline_config['canny_result_dir'])

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)