from time import get_clock_info
from flask import Flask, request
from google.cloud import storage
import os, logging, gcloud_storage


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


# @app.errorhandler(500)
# def server_error(e):
#     logging.exception('An error occurred during a request.')
#     return """
#     An internal error occurred: <pre>{}</pre
#     See logs for full stacktrace.
#     """.format(e), 500



if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)