import os
import sys

import boto3
import botocore
from termcolor import cprint

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
CONFIG_DIR = REPO_DIR

USER = os.environ['USER']
PREFIX = f"{USER}-jaynes"
SECURITY_GROUP_NAME = f"{PREFIX}-sg"
INSTANCE_PROFILE_NAME = f"{PREFIX}-worker"
INSTANCE_ROLE_NAME = f"{PREFIX}-role"

AWS_REGIONS = [
    "ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1",
    "ap-southeast-2", "eu-central-1", "eu-west-1", "sa-east-1", "us-east-1",
    "us-east-2", "us-west-1", "us-west-2", ]


def setup_key_pairs(region, key_name):
    ec2_client = boto3.client("ec2", region_name=region, )
    try:
        cprint(f"Creating key pair with name {key_name}...", "blue", end="\r")
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
        cprint(f"Created key pair with name {key_name}...", "green")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'InvalidKeyPair.Duplicate':
            if not query_yes_no("Key pair with name %s exists. Proceed to delete and recreate?" % key_name, "no"):
                sys.exit()
            cprint(f"Deleting existing key pair with name {key_name}", "red", end="\r")
            ec2_client.delete_key_pair(KeyName=key_name)
            cprint(f"Recreating key pair with name {key_name}...", "blue", end="\r")
            key_pair = ec2_client.create_key_pair(KeyName=key_name)
            cprint(f"Recreated key pair {key_name}", "green", end="\r")
        else:
            raise e

    key_pair_folder_path = os.path.join(CONFIG_DIR, ".secrete")
    file_name = os.path.join(key_pair_folder_path, "%s.pem" % key_name)

    cprint(f"Saving key pair {key_name}", "green", end="\r")
    os.makedirs(key_pair_folder_path, exist_ok=True)
    with os.fdopen(os.open(file_name, os.O_WRONLY | os.O_CREAT, 0o600), 'w') as handle:
        handle.write(key_pair['KeyMaterial'] + '\n')
    cprint(f"Saved key pair {key_name}", "green")
    return key_name


def get_subnets_info(region):
    client = boto3.client("ec2", region_name=region, )
    subnets = client.describe_subnets()['Subnets']
    return [n['AvailabilityZone'] for n in subnets]


def query_yes_no(question, default="yes", allow_skip=False):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if allow_skip:
        valid["skip"] = "skip"
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    if allow_skip:
        prompt += " or skip"
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == "__main__":
    [setup_key_pairs(region, f"{USER}-{region}") for region in AWS_REGIONS]
