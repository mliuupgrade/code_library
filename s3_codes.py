
import boto3
import os
import sys
import threading
import io
import pickle

class Upload_ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):

        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write("\r  %s / %s  (%.2f%%)" % (self._seen_so_far, self._size, percentage))
            sys.stdout.flush()

class Download_ProgressPercentage(object):
    def __init__(self, client, bucket, filename):
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):

        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write("\r  %s / %s  (%.2f%%)" % (self._seen_so_far, self._size, percentage))
            sys.stdout.flush()


class s3_helper:
    def __init__(self):
        self.session = boto3.Session(aws_access_key_id=input("Provide the aws_access_key_id"),
                                     aws_secret_access_key=input("Provide the aws_secret_access_key"),
                                     aws_session_token=input("aws_session_token"),
                                     region_name='us-west-2',)
        self.s3 = self.session.client('s3')
        self.bucket = 'upg-decisionsciences-data-usw2-services'

    def download(self, s3_file, file_location):
        """
        Download data from S3:
        s3_file: provide the name of file including the path (don't include the bucket name)
        file_location: provide the local path, make sure to include "/" at the end,
        do not provide file name, it is the same as the one used in s3
        """
        file_name = s3_file.split('/')[-1]
        progress = Download_ProgressPercentage(self.s3, self.bucket, s3_file)
        self.s3.download_file(self.bucket, s3_file, file_location+file_name, Callback=progress)

    def upload(self, file_location, file_name, s3_location):
        """
        Upload data to S3:
        file_location: provide the local path, make sure to include "/" at the end
        file_name: provide the name of the file to be uploaded
        s3_location: Provide the S3 location, do not include bucket name or file name, make sure to include "/" at the end
        """
        self.s3.upload_file(file_location+file_name, self.bucket, s3_location+file_name, Callback=Upload_ProgressPercentage(file_location+file_name))

    def delete(self, s3_file):
        """
        Delete S3 file:
        s3_file: provide the name of the file including the path (don't include the bucket name)
        """
        self.s3.delete_object(Bucket=self.bucket, Key=s3_file)

    def list_files(self, s3_folder):
        """
        List files in an S3 folder:
        s3_folder: provide the full folder name (don't include the bucket name),
        Provide empty string to list all files in the bucket
        """
        for content in self.s3.list_objects(Bucket=self.bucket, Prefix=s3_folder)['Contents']:
            print(content['Key'])

    def upload_from_ram(self, df, file_name, s3_location, file_type):
        """
        Upload a pandas dataframe in RAM directly to S3 and save in csv/pickle format:
        df: pandas dataframe
        file_name: provide the name of the file to be saved
        s3_location: Provide the S3 location, do not include bucket name or file name, make sure to include "/" at the end
        file_type: 'csv' or 'pickle'
        """
        if file_type=='csv':
            with io.StringIO() as csv_buffer:
                df.to_csv(csv_buffer, index=False)
                self.s3.put_object(Bucket=self.bucket, Key=s3_location+file_name, Body=csv_buffer.getvalue())
        elif file_type=='pickle':
            pickle_buffer = pickle.dumps(df)
            self.s3.put_object(Bucket=self.bucket, Key=s3_location+file_name, Body=pickle_buffer)
        else:
            print ('This file type is not supported by the function. Please save to a local location and upload.')


s3 = s3_helper()

s3.list_files('BR1_Results/')

s3.download('BR1_Results/score_dist/br1_score_dist.csv', '/home/user/Downloads/')
