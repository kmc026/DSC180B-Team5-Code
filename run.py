import sys
import json

from src.main import write_ssim


def main(targets):

    test_config = json.load(open("config/config.json"))

    if 'test' in targets:
        
        write_ssim(test_config['actual_image_dir'], test_config['ground_truth_image_dir'], test_config['SSIM_result_dir'])


if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)