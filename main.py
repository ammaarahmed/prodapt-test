from typing import List
from flask import Flask, request
from google.cloud import storage
import logging, time, gcloud_storage

from google.protobuf import message

from google.protobuf.timestamp_pb2 import Timestamp

from google.cloud.storage import bucket
import pandas as pd 
from io import StringIO # if going with no saving csv file


app = Flask(__name__)

# Configure this environment variable via app.yaml
@app.route('/')
def index():
    return """
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
    """


@app.route('/upload', methods=['POST'])
def upload():
    """Process the uploaded file and upload it to Google Cloud Storage."""
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return 'No file uploaded.', 400

    # Create a Cloud Storage client.
    storage_client = storage.Client()

    #create the bucket where we want to store the file 
    bucket_name = 'prodapt-vertex-test'

    # Get the bucket that the file will be uploaded to.
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob and upload the file's content.
    blob = bucket.blob('pred/' +  uploaded_file.filename)

    blob.upload_from_string(
        uploaded_file.read(),
        content_type=uploaded_file.content_type
    )

    # The public URL can be used to directly access the uploaded file via HTTP.
    
    gcloud_storage.create_data_set(blob.public_url, 4)

    response = gcloud_storage.create_batch_prediction_job_bigquery(4)
    print(response)
    result = {'message': 'predictions stored in BQ', 'response': response}

    # return result, 200

    bqUrlOutpt = response["bqUrlOutpt"]
    modelName = response["modelName"]
    dateString = response["startTime"]

    print(dateString)

    datasetId = bqUrlOutpt[5:]

    predStatus: str = gcloud_storage.get_batch_prediction_job(modelName)

    predTableName: str = gcloud_storage.get_prediction_table(datasetId,dateString)

    predTablePath = datasetId + '.' + predTableName

    predResult, predResultProb = gcloud_storage.get_pred_result(predTablePath)

    result = {
        "predResult":predResult,
        "predResultProb": predResultProb,
        "predStatus": predStatus
    }

    return result, 200 


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre
    See logs for full stacktrace.
    """.format(e), 500

@app.route('/json_upload', methods=['POST'])
def json_upload():
    """
    json input: 
        "ph", "FLOAT64"
        "Hardness", "FLOAT64"
        "Solids", "FLOAT64"
        "Chloramines", "FLOAT64"
        "Sulfate", "FLOAT64"
        "Conductivity", "FLOAT64"
        "Organic_carbon", "FLOAT64"
        "Trihalomethanes", "FLOAT64"
        "Turbidity", "FLOAT64"
        "Potability", "INT64"
    sample: 
    {
        "ph": 9.092223456290965,
        "Hardness": 181.10150923612525,
        "Solids": 17978.98633892625,
        "Chloramines": 6.546599974207941,
        "Sulfate": 310.13573752420444,
        "Conductivity": 398.41081338184466,
        "Organic_carbon": 11.558279443446395,
        "Trihalomethanes": 31.997992727424737,
        "Turbidity": 4.075075425430034,
        "Potability": "INT64",
        "predNum": 
    }
    """
    #pull json response and format as datframe
    featureSet = request.get_json(silent=True)
    df = pd.json_normalize(featureSet).drop("predNum", axis=1)
    predNum = featureSet["predNum"]

    #parse json response as csv 
    f = StringIO()
    df.to_csv(f)
    f.seek(0)
    
    
    #create the bucket where we want to store the file 
    bucket_name = 'prodapt-vertex-test'

    # Get the bucket that the file will be uploaded to.
    gcsClient = storage.Client()
    bucket = gcsClient.get_bucket(bucket_name)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(f'pred/water_potability_{str(predNum)}.csv')
    blob.upload_from_file(f, content_type='text/csv')

    # The public URL can be used to directly access the uploaded file via HTTP.
    gcloud_storage.create_data_set(blob.public_url, predNum)
    response = gcloud_storage.create_batch_prediction_job_bigquery(predNum)

    # return result, 200
    bqUrlOutpt = response["bqUrlOutpt"]
    modelName = response["modelName"]
    dateString = response["startTime"]

    predStatus: str = gcloud_storage.get_batch_prediction_job(modelName)
    while predStatus == 'JobState.JOB_STATE_RUNNING':
        print('Job still running')
        time.sleep(15)
        predStatus: str = gcloud_storage.get_batch_prediction_job(modelName)
        

    datasetId = bqUrlOutpt[5:]

    r = gcloud_storage.get_prediction_table(datasetId,dateString)
    

    # ``JOB_STATE_SUCCEEDED``,``JOB_STATE_FAILED``, ``JOB_STATE_CANCELLED``

    try:
        predTablePath: str = datasetId + '.' + r["predTableName"]
        predResult, predResultProb = gcloud_storage.get_pred_result(predTablePath)
        code = 200
    except TypeError: 
        print('Error - prediction not created yet')
    else: 
        predResult, predResultProb = None 
        code = 400

    result = {
        "predResult": predResult,
        "predResultProb": predResultProb,
        "predStatus": predStatus,
        "errorNum": len(r["errorTables"]),
        "startTime": r["timeStamp"]
    }

    return result, code

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)