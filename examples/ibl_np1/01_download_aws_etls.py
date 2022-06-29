import os

import boto3

from one.api import ONE
from pathlib import Path

# aws s3 sync s3://ibl-brain-wide-map-private/data/etls/yass/ /datadisk/Data/spike_sorting/re_datasets/

one = ONE()
PRIVATE_REPO_NAME = 'ibl-brain-wide-map-private'
DATA_DIR = Path(f"/datadisk/Data/spike_sorting/re_datasets")  # YOUR LOCAL PATH
pids = None  # set to None to download all available

pids = ["64d04585-67e7-4320-baad-8d4589fd18f7",  # CA1
        "31d8dfb1-71fd-4c53-9229-7cd48bee07e4"]  # MOS


def download_s3_folder(s3r, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3r.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(str(local_dir), os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)



s3info = one.alyx.rest('data-repository', 'read', id='ibl-brain-wide-map-private')['json']
session = boto3.Session(
    aws_access_key_id=s3info['Access key ID'],
    aws_secret_access_key=s3info['Secret access key'],
)
s3c = session.client('s3')
result = s3c.list_objects(Bucket=s3info['bucket_name'], Prefix='data/etls/yass/', Delimiter='/')
if pids is None:
    pids = []
    for o in result.get('CommonPrefixes'):
        pids.append(Path(o.get('Prefix')).parts[-1])
    print(pids)


session = boto3.Session(
    aws_access_key_id=s3info['Access key ID'],
    aws_secret_access_key=s3info['Secret access key'],
)

s3r = session.resource('s3')

for pid in pids:
    if DATA_DIR.joinpath(pid).exists():
        continue
    else:
        print(f'downloading {pid}...')
        download_s3_folder(s3r, bucket_name=s3info['bucket_name'], s3_folder=f'data/etls/yass/{pid}', local_dir=DATA_DIR.joinpath(pid))
