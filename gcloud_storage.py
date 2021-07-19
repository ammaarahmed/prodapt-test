from google.cloud import bigquery, storage, aiplatform, aiplatform_v1beta1, bigquery_storage
from google import cloud
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from typing import List,Tuple
import re, datetime, pyarrow
import pandas as pd




yourProject = "prodapt-test"

#*********CREATING BUCKETS, BQ TABLES, PRED JOBS****************
def create_bucket_class_location(bucket_name):
    """Create a new bucket in specific location with storage class"""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "COLDLINE"
    new_bucket = storage_client.create_bucket(bucket, location="us")

    print(
        f"Created bucket {new_bucket.name} in {new_bucket.location} with storage class {new_bucket.storage_class}")

    return new_bucket

def upload_blob(bucketName, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The path to your file to upload
    source_file_name = "local/path/to/file"
    # The ID of your GCS object
    destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucketName)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def create_data_set( bucketUrl, predNum, pred = 1):

    client = bigquery.Client()

    yourDataset =  "pred" if pred == 1 else "pred_result"
    yourTableName =  "water_potability_" + str(predNum) 

    try: 
        dataset_id = "{}.your_dataset".format(client.project)

        # Construct a full Dataset object to send to the API.
        dataset = bigquery.Dataset(yourProject +'.' + yourDataset)

        # TODO(developer): Specify the geographic location where the dataset should reside.
        dataset.location = "US"

        # Send the dataset to the API for creation, with an explicit timeout.
        # Raises google.api_core.exceptions.Conflict if the Dataset already
        # exists within the project.
        dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.
        print("Created dataset {}.{}".format(client.project, dataset.dataset_id))

    except cloud.exceptions.Conflict:
        print(f'Dataset {dataset.dataset_id} already exists, not creating.')
    else:
        print(f'Dataset {dataset.dataset_id} successfully created.') 

    tableId = f"{yourProject}.{yourDataset}.{yourTableName}"

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("ph", "FLOAT64"),
            bigquery.SchemaField("Hardness", "FLOAT64"),
            bigquery.SchemaField("Solids", "FLOAT64"),
            bigquery.SchemaField("Chloramines", "FLOAT64"),
            bigquery.SchemaField("Sulfate", "FLOAT64"),
            bigquery.SchemaField("Conductivity", "FLOAT64"),
            bigquery.SchemaField("Organic_carbon", "FLOAT64"),
            bigquery.SchemaField("Trihalomethanes", "FLOAT64"),
            bigquery.SchemaField("Turbidity", "FLOAT64"),
            bigquery.SchemaField("Potability", "INT64")
            ],
        skip_leading_rows=1,
        # The source format defaults to CSV, so the line below is optional.
        source_format=bigquery.SourceFormat.CSV,
        max_bad_records = 1000

    )

    bq_url = f'bq://{yourProject}.{yourDataset}.{yourTableName}'


    load_job = client.load_table_from_uri(
        bucketUrl, tableId, job_config=job_config
    )  # Make an API request.

    load_job.result()  # Waits for the job to complete.

    destination_table = client.get_table(tableId)  # Make an API request.
    print("Loaded {} rows.".format(destination_table.num_rows))

    return bq_url

def create_batch_prediction_job_bigquery(
    predNum: int,
    instances_format: str = "bigquery",
    predictions_format: str = "bigquery",
    model_name:str = "water-model",
    model_id: int = 4580591829593882624,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):

    yourDatasetInput = 'pred'
    yourDatasetOutput = 'pred_result' 
    yourTableName =  "water_potability_" + str(predNum) 

    bq_url_input = f'bq://{yourProject}.{yourDatasetInput}.{yourTableName}'
    bq_url_output = f'bq://{yourProject}.{yourDatasetOutput}'
    model_path = f'projects/{yourProject}/locations/{location}/models/{str(model_id)}'

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform_v1beta1.JobServiceClient(client_options=client_options)
    model_parameters_dict = {}
    model_parameters = json_format.ParseDict(model_parameters_dict, Value())

    batch_prediction_job = {
        "display_name": model_name,
        "model": model_path,
        "model_parameters": model_parameters,
        "input_config": {
            "instances_format": instances_format,
            "bigquery_source": {"input_uri": bq_url_input},
        },
        "output_config": {
            "predictions_format": predictions_format,
            "bigquery_destination": {"output_uri": bq_url_output},
        },
        # optional
        "generate_explanation": True,
    }

    parent = f"projects/{yourProject}/locations/{location}"
    response = client.create_batch_prediction_job(
        parent=parent, batch_prediction_job=batch_prediction_job
    )


    print(response.output_config.bigquery_destination.output_uri)
    print(response.create_time)
    print(response.name)


    

    return {
        'bqUrlOutpt': response.output_config.bigquery_destination.output_uri,
        'startTime': response.create_time,
        'modelName': response.name   
    }


# def get_last_bucket_num(bucketName: str):
    
    # numReTest = '(\d+)\D*\Z'
    # errorText = 'error-1'

    # result = re.search(numReTest, errorText).group(1)


#     client = storage.Client()
#     for blob in client.list_blobs('bucketname', prefix='abc/myfolder'):
#         print(str(blob))
#     return errorNum, predNum

#******* PULLING BUCKETS, BQ TABLES, PREDICTIONS, STATUS OF PRED ***************

def change_glcoud_str_to_bq_table_format(
    dateString: str, 
    format1: str = '%a, %d %b %Y %H:%M:%S %Z' #recieved format
        ) -> str:
    """takes the date string that is return from the prediction response 
    and reformats it into the big query date format """
    
    format2 = '%Y_%m_%dT%H_%M_%S' #format we want returned
    return datetime.datetime.strptime(dateString, format1).strftime(format2)

def change_gcloud_datetime_to_bq_table_format(startDateTimeObj) -> str:
     return startDateTimeObj.strftime("%Y_%m_%dT%H_%M_%S_%fZ")



def hour_time_shift(timeObject: str, hourShift: int = 0, millTime = False) -> str:
    """works for string objects that are organzied using the following 
    format (the format that the predictions are retuned in ):

    YYYY_MM_DDThh_mm_ss_sssZ

    """

    currentHours: int = int(timeObject[11:13])

    adjustedHours: int = currentHours + hourShift

    if adjustedHours > 12 and millTime != True:
        adjustedHours: str = str(adjustedHours - 12)
    else:
        adjustedHours: str = str(adjustedHours)
    
    if len(adjustedHours) < 2: 
            adjustedHours: str = '0' + adjustedHours  

    return timeObject[:11] + adjustedHours + timeObject[13:]

def get_all_tables_in_dataset(datasetId: str) -> List: 
    """
    here dataset id is simply just the the project name 
    and the dataset under it that we ant to query 
    """
    
    #create client and get dataset
    client = bigquery.Client()
    dataset = client.get_dataset(datasetId)

    #return list of all tables in dataset
    return list(client.list_tables(dataset))


#TODO can get this done without iteration -> try pred_datetime catch error_datetime
def get_prediction_table(datasetId: str, timeStamp: str) -> str: 
    """
    takes the response from
    """

    #reformat time stamps 
    timeStamp = change_gcloud_datetime_to_bq_table_format(timeStamp)
    timeStamp = hour_time_shift(timeStamp, -7)

    #get all tables in requested dataset 
    tables = get_all_tables_in_dataset(datasetId)

    
    #
    errorTable: str = 'errors_'
    predictionTable: str = 'predictions_'
    errorTables: List = []

    #parse through all tables to find one with matching timestamp 
    for table in tables:
        tableID: str = table.table_id
        print(tableID)
        if re.search(errorTable, tableID):
            tableTimeStamp: str = tableID[7:26]
            if tableTimeStamp == timeStamp:
                errorTables.append(table.table_id)
                
        elif re.search(predictionTable, tableID):
            tableTimeStamp: str = tableID[12:31]
            if tableTimeStamp == timeStamp:
                print('SUCCESS')
                break
        else: 
            pass 
    
    return table.table_id

def get_batch_prediction_job(
    modelNameResp: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
) -> str:

    batch_prediction_job_id = re.search('(\d+)\D*\Z', modelNameResp).group(1)
    project = re.search('projects/(\d+)', modelNameResp).group(1)


    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    name = client.batch_prediction_job_path(
        project=project, location=location, batch_prediction_job=batch_prediction_job_id
    )
    response = client.get_batch_prediction_job(name=name)
    print("state:", str(response.state))
    
    return str(response.state)

def get_pred_result(predTablePath: str) -> Tuple[int, float]: 
    """"
    predTablePath: full path to the table that has the prediction results 
    """
    client = bigquery.Client()
    storageclient = bigquery_storage.BigQueryReadClient()

    # Download query results.
    query_string: str = f"""
    SELECT *
    FROM `{predTablePath}`
    """

    print(query_string)

    dataframe = (
        client.query(query_string)
        .result()
        .to_dataframe(bqstorage_client=storageclient)
    )

    predCol = pd.json_normalize(dataframe['predicted_Potability'])

    if predCol['scores'][0][0] >= predCol['scores'][0][1]: 
        predResult: int = predCol['classes'][0][0]
        predResultProb: float =  predCol['scores'][0][0]
    else: 
        predResult: int = predCol['classes'][0][1]
        predResultProb: float =  predCol['scores'][0][1]

    print('Pred: ', predResult, '\n Pred Confidence: ', predResultProb)

    return predResult, predResultProb