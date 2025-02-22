from google.cloud import storage
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import signal

class GCSFolderUploader:
    """
    Handles multi-threaded uploading of a local folder to Google Cloud Storage.
    """

    def __init__(self, bucket_name, credentials_path=None, project_id=None, max_workers=None, chunk_size=None):
        """
        Initializes the GCSFolderUploader.

        Args:
            bucket_name: The name of the GCS bucket.
            credentials_path: (Optional) Path to the service account key file (JSON).
                              If None, attempts to use Application Default Credentials.
            project_id: (Optional) The Google Cloud project ID.  Required if not
                              using Application Default Credentials *and* it's not
                              included in the credentials file.
            max_workers: (Optional) Maximum number of threads for uploads.
            chunk_size: (Optional) Chunk size in MB for uploads.  Defaults to a reasonable size.
                         This can help with large files and network reliability.
        """
        if credentials_path:
            self.client = storage.Client.from_service_account_json(credentials_path, project=project_id)
        elif project_id:
            self.client = storage.Client(project=project_id)
        else:
            self.client = storage.Client()

        self.bucket = self.client.bucket(bucket_name)
        self.max_workers = max_workers or os.cpu_count() * 5  # More workers for I/O bound
        self._thread_local = threading.local()
        self.chunk_size = chunk_size * 1024 * 1024 if chunk_size else None # Convert MB to bytes

        self._stop_event = threading.Event() # For graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler) # Handle termination signals



    def _get_client(self):
        """Gets a thread-local GCS client."""
        if not hasattr(self._thread_local, "client"):
            if hasattr(self, 'credentials_path') and self.credentials_path:
                self._thread_local.client = storage.Client.from_service_account_json(
                    self.credentials_path, project=self.client.project
                )
            elif hasattr(self, 'client') and self.client.project:
                 self._thread_local.client = storage.Client(project=self.client.project)
            else:
                self._thread_local.client = storage.Client()
        return self._thread_local.client


    def _upload_file(self, local_file_path, destination_blob_name):
        """Uploads a single file to GCS with resumable upload support and retry logic."""

        #Use a retry loop in case of transient network errors.
        max_retries = 5
        retry_delay = 5 #seconds

        for attempt in range(max_retries):
            if self._stop_event.is_set():
                return False
            try:
                client = self._get_client()
                bucket = client.bucket(self.bucket.name)
                blob = bucket.blob(destination_blob_name)

                # Configure resumable upload with a specified chunk size.
                if self.chunk_size:
                    blob.chunk_size = self.chunk_size

                blob.upload_from_filename(local_file_path)  #This handles resumable uploads automatically
                print(f"File {local_file_path} uploaded to {destination_blob_name}.")
                return True
            except Exception as e:
                print(f"Error uploading {local_file_path} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to upload {local_file_path} after multiple retries.")
                    return False
        return False

    def _signal_handler(self, signum, frame):
        """Handles signals (SIGINT, SIGTERM) for graceful shutdown."""
        print("Signal received, shutting down...")
        self._stop_event.set()


    def upload_folder(self, local_folder_path, destination_blob_prefix):
        """
        Uploads a local folder to Google Cloud Storage using multi-threading,
        resumable uploads, and retry logic.
        """

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for root, _, files in os.walk(local_folder_path):
                if self._stop_event.is_set():
                    break #Stop creating tasks if shutdown signal
                for file in files:
                    if self._stop_event.is_set():
                        break #Stop creating tasks if shutdown signal
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_folder_path)
                    destination_blob_name = os.path.join(destination_blob_prefix, relative_path)

                    future = executor.submit(self._upload_file, local_file_path, destination_blob_name)
                    futures.append(future)

            # Wait for uploads, and check for errors or shutdown signal.
            for future in as_completed(futures):
                if self._stop_event.is_set():
                    print("Cancelling remaining uploads...")
                    for f in futures: #Cancel other pending uploads.
                        f.cancel()
                    break # Exit the loop if shutdown signal
                if not future.result():
                    print("WARNING: Some files failed to upload.")


# --- Example Usage ---
if __name__ == "__main__":
    bucket_name = "llm_healthcare"
    credentials_path = "credentials.json"  # Or None for ADC
    project_id = "august-bucksaw-436619-e3" # Or None

    local_folder_to_upload = "my_local_folder"
    gcs_upload_prefix = "test"

    # Create a dummy folder with some larger files for testing
    if not os.path.exists(local_folder_to_upload):
        os.makedirs(local_folder_to_upload)
        with open(os.path.join(local_folder_to_upload, "file1.txt"), "wb") as f:
            f.write(b"This is a larger file.\n" * 1024 * 100)  # ~100KB file
        os.makedirs(os.path.join(local_folder_to_upload, "subdir"))
        with open(os.path.join(local_folder_to_upload, "subdir", "file2.dat"), "wb") as f:
            f.write(os.urandom(1024 * 1024 * 5))  # 5MB file
        with open(os.path.join(local_folder_to_upload, "file3.txt"), "wb") as f:
            f.write(b"Another test file.\n" * 512 * 50)

    # Instantiate the uploader (with optional chunk size for testing)
    uploader = GCSFolderUploader(bucket_name, credentials_path=credentials_path, project_id=project_id, max_workers=8, chunk_size=2) # 2 MB chunk

    try:
        uploader.upload_folder(local_folder_to_upload, gcs_upload_prefix)
    except KeyboardInterrupt:
        print("Upload interrupted by user.")

    print("Done!")