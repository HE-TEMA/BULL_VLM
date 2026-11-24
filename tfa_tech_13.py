# Copyright © 2026 Bull® SAS. All rights reserved

"""Image visual question answering and caption similarity service.

This module provides a FastAPI application for processing image-related
notifications. It generates image captions using a Florence-2 vision-language
model, evaluates text-to-caption similarity using CLIP, stores processed
results in MinIO, and publishes resulting metadata to an NGSI-LD context
broker.
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, CLIPTokenizer, CLIPModel
import pandas as pd
import requests
from requests.exceptions import RequestException



from minio import Minio
from minio.error import S3Error
import requests
import json
from fastapi import FastAPI, Request
import random
from pydantic import BaseModel, HttpUrl, Field, ValidationError
from typing import List
import os
import ast
from typing import Optional
import datetime
from tqdm import tqdm





app = FastAPI()
BASE_PATH = os.environ.get("BASE_PATH", "/")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


if DEBUG:
    print("Verbose mode enabled: Debugging information will be shown.")

main_entity = "urn:ngsi-ld:tema:ATOS:TFA-13:ImageVQA:"
counter = 0

euclidean_distance = 0.22


# Function to increment the counter
def increment_counter():
    """Increment the global entity identifier counter by one."""
    global counter
    counter += 1


# Function to get the current value of the counter
def get_counter():
    """Return the current global entity identifier counter.

    Returns:
        int: The current counter value.
    """
    global counter
    return counter


def get_session():
    """Generate a timestamp string for the current local time.

    Returns:
        str: The current timestamp formatted as YYYYMMDDHH.
    """
    return datetime.datetime.now().strftime("%Y%m%d%H")

def create_unique_id():
    """Generate a unique entity identifier using the namespace and timestamp.

    Increments the global counter and combines it with the configured
    entity namespace and the current timestamp.

    Returns:
        str: The generated entity identifier.
    """
    increment_counter()
    session_id = get_session()
    return f"{main_entity}_{session_id}_{get_counter()}"


class MinioClient:
    """Provide file and bucket operations for MinIO object storage.

    Attributes:
        client: Initialized MinIO client used to access object storage.
    """

    def __init__(self, endpoint, access_key, secret_key, secure=True):
        """Initialize the MinIO client.

        Args:
            endpoint (str): MinIO server hostname and port.
            access_key (str): Access key used for authentication.
            secret_key (str): Secret key used for authentication.
            secure (bool): Whether to use HTTPS for the connection.
        """
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

    def create_bucket(self, bucket_name):
        """Create a bucket if it does not already exist.

        Args:
            bucket_name (str): Name of the bucket to create.

        Note:
            MinIO errors are printed rather than propagated to the caller.
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' created successfully.")
            else:
                print(f"Bucket '{bucket_name}' already exists.")
        except S3Error as e:
            print(f"Error creating bucket: {e}")

    def upload_file(self, bucket_name, object_name, file_path):
        """Upload a local file to a MinIO bucket.

        Args:
            bucket_name (str): Destination bucket name.
            object_name (str): Destination object key.
            file_path (str): Path to the local file.

        Returns:
            bool: True if the upload succeeds, otherwise False.
        """
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(
                f"File '{file_path}' uploaded to '{bucket_name}/{object_name}' successfully."
            )
            return True
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False

    def download_file(self, bucket_name, object_name, file_path):
        """Download a MinIO object to a local file.

        Args:
            bucket_name (str): Source bucket name.
            object_name (str): Object key to download.
            file_path (str): Local destination path.

        Returns:
            bool: True if the download succeeds, otherwise False.
        """
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
            print(
                f"File '{object_name}' downloaded from '{bucket_name}' to '{file_path}' successfully."
            )
            return True
        except S3Error as e:
            print(f"Error downloading file: {e}")
            return False

    def list_objects(self, bucket_name):
        """Print the names of objects stored in a bucket.

        Args:
            bucket_name (str): Name of the bucket to inspect.

        Note:
            MinIO errors are printed rather than propagated to the caller.
        """
        try:
            objects = self.client.list_objects(bucket_name)
            print(f"Objects in bucket '{bucket_name}':")
            for obj in objects:
                print(obj.object_name)
        except S3Error as e:
            print(f"Error listing objects: {e}")

class VQA:
    """Generate image captions and visual question-answering results.

    Uses a pretrained Florence-2 model and its associated processor to
    perform image understanding tasks, including caption generation,
    object detection, OCR, and region-based descriptions.

    Attributes:
        model_path (str): Path or identifier of the pretrained model.
        device (str): Device used for inference, either CUDA or CPU.
        torch_dtype (torch.dtype): Floating-point precision used by the model.
        model: Loaded causal language model.
        processor: Processor used to prepare images and text and decode results.
    """
    def __init__(self,model_path):
        """Load the vision-language model and its processor.

        Args:
            model_path (str): Local path or model identifier for the
                pretrained model. Model files must be available locally.
        """
        self.model_path = model_path

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, trust_remote_code=True, local_files_only=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)

    def process_image(self, prompt, image, text_input=None):
        """Run an image-understanding task and post-process the generated output.

        Combines the task prompt with optional additional text, prepares the
        image and text inputs, generates a response, and decodes the output
        using the processor's task-specific post-processing functionality.

        Args:
            prompt (str): Florence-2 task token, such as ``<CAPTION>``,
                ``<DETAILED_CAPTION>``, or ``<OCR>``.
            image (numpy.ndarray): Input image in the format expected by
                the processor.
            text_input (str, optional): Additional text appended to the
                task prompt, such as a phrase to ground in the image.

        Returns:
            dict or other processor-defined result: Task-specific output
            returned by the processor's post-processing method.
        """
        if text_input is None:
            prompt_c = prompt
        else:
            prompt_c = prompt + text_input

        height, width, channels = image.shape

        inputs = self.processor(text=prompt_c, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(width, height))

        return parsed_answer

    def process_image_score(self,prompt, image,text_input =None):
        """Run an image-understanding task and return score-aware results.

        Generates output with beam-search scores enabled and calculates
        transition scores for the generated sequences. The resulting sequence
        and scores are passed to the processor for task-specific parsing.

        Args:
            prompt (str): Florence-2 task token, such as ``<OD>`` or
                ``<OPEN_VOCABULARY_DETECTION>``.
            image (numpy.ndarray): Input image in the format expected by
                the processor.
            text_input (str, optional): Additional text appended to the
                task prompt, such as a target object description.

        Returns:
            dict or other processor-defined result: Parsed task output
            produced by the processor's post-processing method.
        """
        if text_input is None:
            prompt_c = prompt
        else:
            prompt_c = prompt + text_input

        height, width, channels = image.shape

        inputs = self.processor(text=prompt_c, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
  
        generated_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        return_dict_in_generate=True,
        output_scores=True,
        )
        generated_text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=False)[0]

        prediction, scores, beam_indices = generated_ids.sequences, generated_ids.scores, generated_ids.beam_indices
        transition_beam_scores = self.model.compute_transition_scores(
            sequences=prediction,
            scores=scores,
            beam_indices=beam_indices,
        )

        parsed_answer = self.processor.post_process_generation(sequence=generated_ids.sequences[0], 
            transition_beam_score=transition_beam_scores[0],
            task=prompt, image_size=(width, height)
        )

        return parsed_answer


class CLIP_model:
    """Calculate semantic similarity between two text descriptions using CLIP.

    Uses the pretrained CLIP ViT-B/32 text encoder to generate text embeddings
    and compares their normalized representations using cosine similarity.

    Attributes:
        tokenizer: CLIP tokenizer used to encode text.
        model: Pretrained CLIP model used to extract text features.
    """

    def __init__(self):
        """Load the pretrained CLIP tokenizer and text-image model."""
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_similarity(self, text1, text2):
        """Calculate cosine similarity between two text descriptions.

        Tokenizes both descriptions, extracts their text embeddings, normalizes
        the embeddings, and calculates cosine similarity.

        Args:
            text1 (str): First text description, typically the original caption
                or social media post text.
            text2 (str): Second text description, typically the generated
                image caption.

        Returns:
            torch.Tensor: Cosine similarity tensor for the two descriptions.
                The value is generally between -1 and 1.
        """
        inputs1 = self.tokenizer(
            text1, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs2 = self.tokenizer(
            text2, return_tensors="pt", padding=True, truncation=True, max_length=77
        )

        with torch.no_grad():
            outputs1 = self.model.get_text_features(**inputs1)
            outputs2 = self.model.get_text_features(**inputs2)

        outputs1 = outputs1 / outputs1.norm(dim=-1, keepdim=True)
        outputs2 = outputs2 / outputs2.norm(dim=-1, keepdim=True)

        similarity = torch.nn.functional.cosine_similarity(outputs1, outputs2)
        return similarity

class PropertyValue(BaseModel):
    """Represent a typed property containing a string value.

    Attributes:
        type (str): NGSI-LD property type.
        value (str): Property value.
    """
    type: str
    value: str

class SocialMediaImages(BaseModel):
    """Validate metadata for a social media image collection or post.

    Attributes:
        id (str): Identifier of the source entity.
        entity_type (str): Entity type supplied through the ``type`` alias.
        creator (PropertyValue): Entity creator metadata.
        title (PropertyValue): Title metadata.
        description (PropertyValue): Description metadata.
        bucket (PropertyValue): MinIO bucket metadata.
        filename (PropertyValue): Source object name.
        minio_url (PropertyValue): URL of the stored source object.
        bm_id (PropertyValue): Associated business mission identifier.
        alert_ref (PropertyValue, optional): Optional alert reference.
        resource_creation (PropertyValue): Resource creation metadata.
    """
    id: str
    entity_type: str = Field(..., alias="type")
    creator: PropertyValue
    title: PropertyValue
    description: PropertyValue
    bucket: PropertyValue
    filename: PropertyValue
    minio_url: PropertyValue
    bm_id: PropertyValue
    alert_ref: Optional[PropertyValue] = None
    resource_creation: PropertyValue

    class Config:
        """Configure input aliases and handling of extra fields."""
        allow_population_by_field_name = True
        extra = "ignore"

class NotificationWrapper(BaseModel):
    """Validate an incoming notification containing social media image entities.

    Attributes:
        id (str): Notification identifier.
        notif_type (str): Notification type supplied through the ``type`` alias.
        subscriptionId (str): Identifier of the subscription that generated
            the notification.
        notifiedAt (str): Timestamp associated with the notification.
        data (List[SocialMediaImages]): Image entities included in the
            notification.
    """
    id: str
    notif_type: str = Field(..., alias="type")  # "Notification"
    subscriptionId: str
    notifiedAt: str
    data: List[SocialMediaImages]



def createOrionEntity(orion_ld_endpoint, entity):
    """Create an NGSI-LD entity in the configured context broker.

    Serializes the supplied entity as JSON and sends it to the broker's
    entity creation endpoint. The request includes JSON-LD content negotiation
    headers.

    Args:
        orion_ld_endpoint (str): Base URL of the NGSI-LD context broker.
        entity (dict): Entity payload to submit.

    Returns:
        bool: True if the request completes with a successful HTTP status
        code; False if the request fails.
    """
    h = {
        "Content-Type": "application/json",
        "Accept": "application/ld+json",
        "Link": '<https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    # {"content-type": "application/json"}
    try:
        response = requests.post(
            url=f"{orion_ld_endpoint}/ngsi-ld/v1/entities",
            headers=h,
            data=json.dumps(entity),
        )
        response.raise_for_status()  # Raise an error for bad status codes
        print(f"Entity created successfully with status code: {response.status_code}")
        return True
    except requests.RequestException as e:
        print(f"Failed to create entity: {e}")
        return False


@app.post(f"{BASE_PATH}/test")
# @app.post("/test")
async def test():
    """Provide a basic endpoint for testing service availability."""
    print("test")

@app.post(f"{BASE_PATH}/json")
async def json_test():
    """Initialize caption and similarity columns in a local JSON dataset.

    Reads the configured social media image dataset into a pandas DataFrame,
    creates empty caption and similarity columns, and writes the modified
    dataset to a JSON output file.
    """
    df = pd.read_json('photo_posts_13_20251016104351.json', dtype={'id': str})
    df['caption'] = ""        
    df['similarity'] = 0.0     # initialize as float
    df.to_json('test_output.json', orient='records', lines=True)



@app.post(f"{BASE_PATH}/image")
async def image_test():
    """Demonstrate the available Florence-2 image-understanding tasks.

    Processes predefined local images using caption generation, phrase
    grounding, object detection, open-vocabulary detection, dense-region
    captioning, region proposals, and OCR. It also calculates CLIP similarity
    scores for selected captions and saves an annotated image containing
    detected bounding boxes.
    """
    image = cv2.imread("images/cat.jpg")

    input_txt= "an image of a grey and white tabby cat"

    print("Caption: ")
    prompt = "<CAPTION>"
    response = vqa.process_image(prompt, image)
    print(response)


    raw_text = response['<CAPTION>']
    similarity = clipM.get_similarity(input_txt, raw_text)
    print(f"similarity: {similarity}")
    print()


    print("Detailed caption: ")
    prompt = "<DETAILED_CAPTION>"
    response = vqa.process_image(prompt, image)
    print(response)

    raw_text = response['<DETAILED_CAPTION>']
    similarity = clipM.get_similarity(input_txt, raw_text)
    print(f"similarity: {similarity}")

    print()

    print("More detailed caption: ")
    prompt = "<MORE_DETAILED_CAPTION>"
    response = vqa.process_image(prompt, image)
    print(response)

    raw_text = response['<MORE_DETAILED_CAPTION>']
    similarity = clipM.get_similarity(input_txt, raw_text)
    print(f"similarity: {similarity}")
    print()

    print("Caption to phrase grounding: ")
    prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    response = vqa.process_image(prompt, image, text_input="Cat's eye")
    print(response)
    print()

    print("Object detection: ")
    prompt = "<OD>"
    response = vqa.process_image_score(prompt, image)
    print(response)
    print()

    print("Open vocabulary detection: ")
    prompt = "<OPEN_VOCABULARY_DETECTION>"
    response = vqa.process_image_score(prompt, image, text_input="cat eye")
    print(response)
    print()
    
    bboxes = response['<OPEN_VOCABULARY_DETECTION>']['bboxes']
    labels = response['<OPEN_VOCABULARY_DETECTION>']['bboxes_labels']

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Put label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Save or show image
    cv2.imwrite('images/output_with_boxes.jpg', image)

    print("Dense region caption: ")
    prompt = "<DENSE_REGION_CAPTION>"
    response = vqa.process_image(prompt, image)
    print(response)
    print()

    print("Region proposal: ")
    prompt = "<REGION_PROPOSAL>"
    response = vqa.process_image(prompt, image)
    print(response)
    print()

    image = cv2.imread("images/hello_world.jpg")

    print("Region proposal: ")
    prompt = "<OCR>"
    response = vqa.process_image(prompt, image)
    print(response)
    print() 

    print("Region proposal: ")
    prompt = "<OCR_WITH_REGION>"
    response = vqa.process_image(prompt, image)
    print(response)
    print()   


@app.post(f"{BASE_PATH}/notification")
async def json_input(subs: NotificationWrapper):
    """Process social media image records and generate descriptive captions.

    Downloads the notification's JSON dataset from MinIO, retrieves each
    referenced image, generates a detailed caption using Florence-2, and
    calculates semantic similarity between the source text and generated
    caption using CLIP.

    The updated dataset is saved locally and uploaded to MinIO. When the
    upload succeeds, an NGSI-LD entity describing the processed output is
    submitted to the context broker.

    Args:
        subs (NotificationWrapper): Validated notification containing source
            dataset metadata.

    Returns:
        None

    Note:
        This endpoint processes the first entity in the notification's
        data list. Individual image request errors are recorded in the
        output dataset when they are caught by the request-error handler.
    """
    entity = subs.data[0]  # grab the first entity

    object_name_download = entity.filename.value
    bucket_name_download = entity.bucket.value
    business_mission = entity.bm_id

    print(entity)

    ret = minio_client.download_file(bucket_name_download, object_name_download, "dummy.json")

    basename = os.path.basename(object_name_download)
    object_name2 = os.path.splitext(os.path.basename(basename))[0] + "_cap.json"

    print(f"download ok {ret}")

    if ret == True:
    
        #start json input
        df = pd.read_json('dummy.json', dtype={'id': str})
    

        # Add empty columns to hold caption and similarity values
        df['caption'] = ""        
        df['similarity'] = 0.0     # initialize as float

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        #for index, row in df.iterrows():
            
            tweet_id = row['id']
            text = row['text']
            photo_url = row['photo_url']

            if DEBUG:
                print(f"ID: {tweet_id}")
                print(f"Text: {text}")
                print(f"Photo URL: {photo_url}")

            try:

                req_response = requests.get(photo_url)

                if req_response.status_code == 200:
                    # Convert image bytes to NumPy array
                    img_array = np.frombuffer(req_response.content, dtype=np.uint8)

                    # Decode to OpenCV image
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    prompt = "<MORE_DETAILED_CAPTION>"
                    response = vqa.process_image(prompt, img)
                    caption = response['<MORE_DETAILED_CAPTION>']

                    similarity = clipM.get_similarity(text, caption)
                    similarity = similarity.item()
                    # Save them back into the DataFrame
                    df.at[index, 'caption'] = caption
                    df.at[index, 'similarity'] = similarity

                    if DEBUG:
                        print(f"caption: {caption}")
                        print(f"similarity: {similarity}")
                    
            except RequestException as e:
                df.at[index, 'caption'] = "caption not available"
                df.at[index, 'similarity'] = -1.0
                
                if DEBUG:
                    print(f"Skipping URL due to request error: {photo_url}\nError: {e}")
                    print(f"caption: caption not available")
                    print(f"similarity: -1.0")
            
            if DEBUG:
                print("-" * 40)

        df.to_json('updated_file.json', orient='records', lines=True)

        success = minio_client.upload_file(bucket_name, object_name2, 'updated_file.json')

        if success:

                current_id = create_unique_id()
                

                entity = {
                    "id": current_id,
                    "type": os.environ.get("BROKER_TYPE_ID"),  # "ImageVQA",
                    "bucket": f"{bucket_name}",
                    "creator": "ATOS",
                    "filename": f"{object_name2}",
                    "minio_url": f"https://{endpoint}/{bucket_name}/{object_name2}",
                }
                print(entity)

                entity_created = createOrionEntity(orion_ld_endpoint, entity)
                if entity_created:
                    print("NGSI-LD entity creation succeeded")
                else:
                    print("NGSI-LD entity creation failed")


if __name__ == "__main__":
    endpoint = os.environ.get("endpoint")
    access_key = os.environ.get("access_key")
    secret_key = os.environ.get("secret_key")
    bucket_name = "atos"
    object_name = "example-object"

    orion_ld_endpoint = (
        os.environ.get("BROKER_URL")
    )

    minio_client = MinioClient(endpoint, access_key, secret_key, secure=True)

    vqa = VQA("microsoft/Florence-2-large")
    clipM = CLIP_model()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT")))
