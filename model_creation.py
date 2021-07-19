import requests, csv 
import pandas as pd
from google.cloud import bigquery, storage, aiplatform

from google import cloud

# from google.cloud import storage



csv_url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'

bucket_url ='gs://prodapt-vertex-test/water_potability.csv'

#********************************************
#with requests 
req = requests.get(csv_url)
text = req.iter_lines()
reader = csv.reader(text, delimiter=',')
print(type(reader))

#with pandas 
data = pd.read_csv(csv_url)
print(data.dtypes)
print(data.columns)


#********************************************


#**********CREATING BUCKETS
# # Imports the Google Cloud client library
# from google.cloud import storage

# # Instantiates a client
# storage_client = storage.Client()

# # The name for the new bucket
# bucket_name = "my-new-bucket"

# # Creates the new bucket
# bucket = storage_client.create_bucket(bucket_name)

# print("Bucket {} created.".format(bucket.name))

#*********ADDING TO BUCKETS 

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


#****************DOWNLOADING TO BUCKETS 

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


#***************CONNECTING TO BIG QUERY 
# export GOOGLE_APPLICATION_CREDENTIALS="/mnt/c/Users/ammar.ahmed/Desktop/prodapt/vertex-final/prodapt-test-3043986cc053.json"

# Construct a BigQuery client object.
client = bigquery.Client()

yourProject = "prodapt-test"
yourDataset =  "water_kaggle"
yourTableName =  "water_potability"

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


# TODO(developer): Set table_id to the ID of the table to create.
table_id = f"{yourProject}.{yourDataset}.{yourTableName}"

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

load_job = client.load_table_from_uri(
    bucket_url, table_id, job_config=job_config
)  # Make an API request.

load_job.result()  # Waits for the job to complete.

destination_table = client.get_table(table_id)  # Make an API request.
print("Loaded {} rows.".format(destination_table.num_rows))



bq_url = f'bq://{yourProject}.{yourDataset}.{yourTableName}'
vertex_loc = 'us-central1'

aiplatform.init(project=yourProject, location=vertex_loc)

dataset = aiplatform.TabularDataset.create(
    display_name=yourDataset, bq_source=bq_url,
)


dataset.wait()

print(f'\tDataset: "{dataset.display_name}"')
print(f'\tname: "{dataset.resource_name}"')

datasetLoc = dataset.resource_name

dataset = aiplatform.TabularDataset(datasetLoc)

job = aiplatform.AutoMLTabularTrainingJob(
  display_name=yourDataset,
  optimization_prediction_type="classification",
  optimization_objective="maximize-au-roc",
)

model = job.run(
    dataset=dataset,
    target_column="Potability",
    training_fraction_split=0.6,
    validation_fraction_split=0.2,
    test_fraction_split=0.2,
    budget_milli_node_hours=1000,
    model_display_name="my-automl-model",
    disable_early_stopping=False,
)



endpoint = model.deploy(machine_type="n1-standard-4",
                        min_replica_count=1,
                        max_replica_count=5,
                        accelerator_type='NVIDIA_TESLA_K80',
                        accelerator_count=1)


model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri="gs://python/to/my/model/dir",
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-2:latest",
)

model = aiplatform.Model(f'/projects/{yourProject}/locations/us-central1/models/{my-automl-model}')
