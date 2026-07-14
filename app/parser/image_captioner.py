from PIL import Image
from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration

import logging

logger = logging.getLogger(__name__)

processor = None
model = None

def load_blip():

    global processor
    global model

    try:

        if processor is None:

            logger.info(
                "Loading BLIP processor"
            )

            processor = (
                BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
            )

        if model is None:

            logger.info(
                "Loading BLIP model"
            )

            model = (
                BlipForConditionalGeneration
                .from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
            )

        logger.info(
            "BLIP loaded successfully"
        )

        return True

    except Exception as e:

        logger.exception(
            f"BLIP loading failed: {str(e)}"
        )

        return False


def generate_image_caption(image_path):

    try:
        
        if not load_blip():

            return (
                "Image caption unavailable"
            )

        logger.info(
            f"Generating caption for image: {image_path}"
        )

        with Image.open(image_path) as raw_image:

            image = raw_image.convert(
                "RGB"
            )

            inputs = processor(
                image,
                return_tensors="pt"
            )

        output = model.generate(
            **inputs,
            max_new_tokens=50
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption

    except Exception as e:

        logger.exception(
            f"Image caption generation failed: {str(e)}"
        )

        return "Unable to describe image"