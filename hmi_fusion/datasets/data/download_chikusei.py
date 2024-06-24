import os
import boto3
os.environ['aws_access_key_id'] = 'K2KBLMB43B4SQV0JA1AJ'
os.environ['aws_secret_access_key'] = 'wZ9AoBbAEvIXPp3jjSNtzoq7vgZPrdEqQmKMP2ha'

s3 = boto3.resource('s3')
# Print out bucket names
for bucket in s3.buckets.all():
    print(bucket.name)