import os

import boto3
import botocore
from termcolor import cprint
from tqdm import tqdm

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


def remove_instance_profile():
    iam = boto3.resource('iam')
    iam_client = boto3.client("iam")

    try:
        iam_client.remove_role_from_instance_profile(
            InstanceProfileName=INSTANCE_PROFILE_NAME,
            RoleName=INSTANCE_ROLE_NAME)
    except:
        pass
    try:
        iam_client.delete_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
    except:
        pass

    try:
        existing_role = iam.Role(INSTANCE_ROLE_NAME)
        existing_role.load()

        for prof in existing_role.instance_profiles.all():
            for role in prof.roles:
                prof.remove_role(RoleName=role.name)
                cprint(f'removing {role.name}', 'green')
            cprint(f'removing {prof}', 'green')
            prof.delete()
        for policy in existing_role.policies.all():
            policy.delete()
        for policy in existing_role.attached_policies.all():
            existing_role.detach_policy(PolicyArn=policy.arn)
        existing_role.delete()
    except:
        pass


def setup_instance_profile():
    iam_client = boto3.client("iam")
    iam = boto3.resource('iam')

    iam_client.create_role(Path='/', RoleName=INSTANCE_ROLE_NAME,
                           AssumeRolePolicyDocument=open("role-policy.json", 'r').read())

    role = iam.Role(INSTANCE_ROLE_NAME)
    role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess')
    role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/ResourceGroupsAndTagEditorFullAccess')

    iam_client.put_role_policy(RoleName=role.name, PolicyName='JaynesWorker',
                               PolicyDocument=open("additional-policies.json", 'r').read())

    iam_client.create_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
    iam_client.add_role_to_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME,
                                            RoleName=INSTANCE_ROLE_NAME)


def setup_security_groups(region):
    ec2 = boto3.resource("ec2", region_name=region, )
    ec2_client = boto3.client("ec2", region_name=region, )

    existing_vpcs = list(ec2.vpcs.all())
    assert len(existing_vpcs) >= 1, f"There is no existing vpc in {region}"
    for vpc in existing_vpcs:
        try:
            security_group, *_ = list(vpc.security_groups.filter(GroupNames=[SECURITY_GROUP_NAME]))
            break
        except:
            pass
    else:
        cprint(f"Creating security group in VPC {vpc.id}", "blue", end="\r")
        security_group = vpc.create_security_group(
            GroupName=SECURITY_GROUP_NAME, Description='Security group for Jaynes')
        ec2_client.create_tags(Resources=[security_group.id],
                               Tags=[dict(Key='Name', Value=SECURITY_GROUP_NAME)])
        cprint(f"Created security group in VPC {vpc.id}", "green")

    try:
        cprint(f"Authorizing Ingress...", "blue", end="\r")
        security_group.authorize_ingress(FromPort=22, ToPort=22, IpProtocol='tcp', CidrIp='0.0.0.0/0')
        security_group.authorize_engress(FromPort=22, ToPort=22, IpProtocol='tcp', CidrIp='0.0.0.0/0')
    except botocore.exceptions.ClientError as e:
        assert e.response['Error']['Code'] == 'InvalidPermission.Duplicate'
    cprint(f"Security group {security_group.id} is created.", "green")

    return security_group.id


def get_subnets_info(region):
    client = boto3.client("ec2", region_name=region, )
    subnets = client.describe_subnets()['Subnets']
    return [n['AvailabilityZone'] for n in subnets]


if __name__ == "__main__":
    import yaml

    all_subnets = sum(map(get_subnets_info, tqdm(AWS_REGIONS)), [])
    with open('ec2_subnets.yml', 'w+') as f:
        yaml.dump(all_subnets, f)

    remove_instance_profile()
    setup_instance_profile()
    sg_ids = list(map(setup_security_groups, AWS_REGIONS))
