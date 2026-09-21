from PIL import Image
import io

def prepare_image(image_bytes: bytes) -> Image.Image:
    """
    Validates and converts raw bytes to a PIL Image ensuring RGB format.
    Raises ValueError on invalid or corrupted images.
    """
    if not image_bytes:
        raise ValueError("Empty image payload received")
        
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Validate image integrity
        
        # Re-open after verify() as PIL requires
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image format: {str(e)}")
