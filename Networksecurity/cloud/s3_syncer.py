import os 

class S3Sync:
    def sync_folder_to_s3(self, folder: str, aws_bucket_url: str):
        # Implement the logic to sync the folder to S3 using AWS CLI or Boto3
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(command)
        
    def sync_folder_from_s3(self, aws_bucket_url: str, folder: str):
        # Implement the logic to sync the folder from S3 using AWS CLI or Boto3
        command = f"aws s3 sync {aws_bucket_url} {folder}"
        os.system(command)