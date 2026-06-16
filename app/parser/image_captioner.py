from PIL import Image
from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration

import logging

logger = logging.getLogger(__name__)

logger.info(
    "Loading BLIP image captioning model"
)

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

logger.info(
    "BLIP model loaded successfully"
)


def generate_image_caption(image_path):

    try:

        logger.info(
            f"Generating caption for image: {image_path}"
        )

        image = Image.open(
            image_path
        ).convert(
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

        logger.info(
            f"Generated caption: {caption}"
        )

        return caption

    except Exception as e:

        logger.exception(
            f"Image caption generation failed: {str(e)}"
        )

        return "Unable to describe image"