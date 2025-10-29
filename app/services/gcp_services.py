from google.cloud import storage
import logging
import os

logger = logging.getLogger(__name__)

def download_blob_if_needed(bucket_name, source_blob_name, destination_file_name):
    """Download a blob from GCP bucket if not present locally."""
    if os.path.exists(destination_file_name):
        logger.info(f"File already exists locally: {destination_file_name}")
        return
        
    # Create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded {source_blob_name} to {destination_file_name}")



