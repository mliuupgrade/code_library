import boto3, sagemakerteam
import time
import pandas as pd
import os
import io
import pickle
from io import StringIO
class RedshiftClient:
    def __init__(self, project_name):
        """
        RedshiftClient is used for interacting with RedShift databases in SageMaker
        project_name: will be used to create a personal folder 
        with the prefix "project_name" in S3 bucket of the team
        "project_name" is the prefix using for your work on this specific project
        """
        # base parameters 
        self.account = 'sagemakerprod'
        self.team = 'decision-sciences'
        
        # s3prefix and s3bucket are used to supply S3 bucket/prefix data
        # The s3bucket is team-wide and belongs to your team's SageMaker domain
        # The prefix holds an individual project
        self.s3prefix = f"{project_name}"
        self.s3bucket = f"upg-sagemaker-{self.team}-usw2-{self.account}"
        self.s3path = f"s3://{self.s3bucket}/{self.s3prefix}"
        
        # Tagging for cost tracking
        self.tags=[{'Key':'upg-env','Value':'sandbox'},{'Key':'upg-app-owner','Value':self.team}]
        # check for typos
        print(f"s3://{self.s3bucket}/{self.s3prefix}")
        print(f"Tagging for cost tracking:{self.tags}")
        
        # get the aws role and region we're running in
        self.role = sagemaker.get_execution_role()
        self.region = boto3.session.Session().region_name
        self.aws_account='179210262644'
        self.redshift_role=f"arn:aws:iam::{self.aws_account}:role/redshift-sagemaker-assume-role-usw2-{self.account}"
        self.role_chain=f"{self.redshift_role},{self.role}"
       
        # create clients for 'redshift' and 'redshift-data' 
        self.redshift = boto3.client('redshift')
        self.redshift_data = boto3.client('redshift-data')
        self.s3_client = boto3.client('s3',region_name=self.region)
        
        self.db_user = 'sagemaker_decision_sciences_app'
        
        self.redshift_creds = self.redshift.get_cluster_credentials(
            DbUser=self.db_user,
            DbName='sagemaker',
            ClusterIdentifier='sagemaker',
            DurationSeconds=3600
        )
        print(self.redshift_creds)
        
    def execute(self, script, timeout=600, wait_time=5):
        query = self.redshift_data.execute_statement(
            ClusterIdentifier='sagemaker',
            Database='sagemaker',
            DbUser=self.db_user,
            Sql=script
        )
        query_id = query['Id']
        start_time = time.time()
        elapsed_time = 0
        describe_statement = self.redshift_data.describe_statement(Id=query_id)
        status = describe_statement['Status']
        # Wait for query to finish running
        while status in ('STARTED', 'SUBMITTED', 'PICKED') and elapsed_time < timeout:
            time.sleep(wait_time)  # wait for 5 seconds before polling again
            describe_statement = self.redshift_data.describe_statement(Id=query_id)
            status = describe_statement['Status']
            elapsed_time = time.time() - start_time
        if elapsed_time >= timeout:
            print(f"Query timed out after {timeout} seconds.")
            return None
        if status == 'FAILED':
            error = describe_statement['Error']
            print(f"Query failed with status: {status} with error: {error}")
            return None
        if status not in ('STARTED', 'SUBMITTED', 'PICKED'):
            print(f"Query status: {status}")
        return query_id
            
    # Read over the socket
    def download(self, sql, timeout=1200):
        # get a pointer to your query (not data)
        query_id = self.execute(sql, timeout, 5)
        if not query_id:
        # Handle the case where query execution failed or timed out
            print("Failed to obtain query_id.")
            return None
        # once the query FINISHED we can get the data
        column_names = [col['label'] for col in self.redshift_data.get_statement_result(Id=query_id)['ColumnMetadata']]
        response = self.redshift_data.get_statement_result(Id=query_id)
        values_only = []
        while 'Records' in response and response['Records']:
            results = response['Records']
            new_values = [[col[list(col.keys())[0]] for col in row] for row in results]
            values_only.extend(new_values)
            if 'NextToken' in response:
                response = self.redshift_data.get_statement_result(Id=query_id, NextToken=response['NextToken'])
            else:
                break
        df = pd.DataFrame(values_only, columns=column_names)
        return df
    
    # Use UNLOAD to get Redshift Data
    def query_unload(self, sql, timeout=1200):
        unload_sql = f"UNLOAD ($${sql}$$) TO '{self.s3path}' IAM_ROLE '{self.role_chain}' FORMAT CSV HEADER EXTENSION 'csv' ALLOWOVERWRITE"
        query_id = self.execute(unload_sql, timeout, 20)
        if not query_id:
        # Handle the case where query execution failed or timed out
            print("Failed to obtain query_id.")
            return None
        # Check the content in the s3path
        bucket_name, prefix = self.s3path.replace('s3://', '').split('/', 1)
        response = self.s3_client.list_objects(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            s3_files = response['Contents']
            print(f"{len(s3_files)} files pushed into S3")
            print(f"Check the data at: {self.s3path}")
        else: 
            print(response)
            print('bucket is empty')
        # Read each CSV file into a DataFrame and store them in a list
        df_list = []
        for obj in response['Contents']:
            file_name = obj['Key']
            if file_name.endswith('.csv'):
                csv_obj = self.s3_client.get_object(Bucket=bucket_name, Key=file_name)
                body = csv_obj['Body']
                csv_string = body.read().decode('utf-8')
                df = pd.read_csv(StringIO(csv_string))
                df_list.append(df)
        # Concatenate all the DataFrames into one
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        return final_df
    
    def upload_from_ram(self, df, file_name, file_type):
        """
        Upload a pandas dataframe in RAM directly to S3 and save in csv/pickle format:
        df: pandas dataframe
        file_name: provide the name of the file to be saved
        file_type: 'csv' or 'pickle'
        """
        if file_type=='csv':
            with io.StringIO() as csv_buffer:
                df.to_csv(csv_buffer, index=False)
                response = self.s3_client.put_object(Bucket=self.s3bucket, Key=f"{self.s3prefix}/{file_name}", Body=csv_buffer.getvalue())
        elif file_type=='pickle':
            pickle_buffer = pickle.dumps(df)
            response = self.s3_client.put_object(Bucket=self.s3bucket, Key=f"{self.s3prefix}/{file_name}", Body=pickle_buffer)
        else:
            print ('This file type is not supported by the function. Please save to a local location and upload.')
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 200:
            print(f"Successful. S3 bucket location: {self.s3bucket}/{self.s3prefix}/{file_name}.{file_type}")
        else:
            print(f"Error: {response.get('Error', {}).get('Message')}")