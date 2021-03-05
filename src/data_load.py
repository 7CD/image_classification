import argparse
import os
import yaml

from torchvision.datasets.utils import download_and_extract_archive


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../params.yaml', help="Path to project config file.", )
    parser.add_argument('--size', choices=['full', '320', '160'], required=True, help='Size if dataset images.')
    return parser.parse_args()


def load_dataset(url, data_dir):
    download_and_extract_archive(url, download_root=data_dir, 
                                 extract_root=data_dir, remove_finished=False)


if __name__ == '__main__':
    args = parse_arguments()
    config = yaml.safe_load(open(args.config))
    url_name = {'full': 'full_size_url', '320': '320px_url', '160': '160px_url'}
    url = config['data_load'][url_name[args.size]]
    data_dir = os.path.join(os.path.dirname(args.config), config['data_load']['data_dir'])
    load_dataset(url, data_dir)
