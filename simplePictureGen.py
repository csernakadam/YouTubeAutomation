import os
import io
import sys
from google import genai
from google.genai import types
from PIL import Image as PillowImage
from datetime import datetime # Import the datetime module

# --- Configuration ---
try:
    # Attempt to initialize the Gemini client
    CLIENT = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
    sys.exit(1)

# Base directory for all outputs
OUTPUT_DIR = "MindsetTales"
# The directory is created if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------

def generate_images(client: genai.Client, topic_prompt: str, num_images: int, output_path: str):
    """
    Uses the Imagen model to generate a specified number of images based on a topic prompt.
    The filename now includes a timestamp to prevent overwriting across different runs.
    """
    if num_images < 1 or num_images > 8:
        print(f"Error: Number of images must be between 1 and 8. Got {num_images}.")
        return

    # --- New Logic for Unique Filename Prefix ---
    # 1. Create a timestamp for a unique run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Create a safe, short prefix from the topic
    topic_slug = "".join(c if c.isalnum() else "_" for c in topic_prompt[:20]).lower().strip("_")

    # 3. Combine the slug and timestamp for the unique file prefix
    base_filename_prefix = f"{topic_slug}_{timestamp}"
    # --- End New Logic ---


    print(f"\n🎨 Step 1: Generating {num_images} image(s) for topic: '{topic_prompt}'...")
    print(f"   Saving results to directory: {output_path}")

    # Define a professional style for the YouTube Short aesthetic
    style_prompt_suffix = (
        "mistic, studio quality, anime picture"
    )

    full_prompt = f"The concept: '{topic_prompt}'. {style_prompt_suffix}"


    try:
        # Use the powerful Imagen 4.0 model
        result = client.models.generate_images(
            model='imagen-4.0-generate-001', #imagen-4.0-ultra-generate-001, imagen-4.0-fast-generate-001
            prompt=full_prompt,
            config=dict(
                number_of_images=num_images,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1",
                image_size="2K",
            )
        )

        if result.generated_images:
            for i, generated_image in enumerate(result.generated_images):
                image_bytes = generated_image.image.image_bytes
                img = PillowImage.open(io.BytesIO(image_bytes))

                # Naming convention: topic_slug_timestamp_01.jpg, topic_slug_timestamp_02.jpg, etc.
                file_path = os.path.join(output_path, f"{base_filename_prefix}_{i + 1:02d}.jpg")
                img.save(file_path)
                print(f"   ✅ Image {i + 1}/{num_images} successfully saved to: {file_path}")

            print("\n🎉 Image Generation Complete!")
        else:
            print("   ❌ Could not generate images. Result object was empty.")

    except Exception as e:
        print(f"   ❌ Error during image generation: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # =======================================================
    # 📝 EDIT THESE VARIABLES to change the generated content!
    # =======================================================
    TARGET_TOPIC = "Create a profile picture for Mindset Tales: Stories that reshape your thinking and transform your life. Each tale is a tool for focus, strength, and growth — a blueprint for breaking old patterns and building a stronger mindset. Change your story. Change your life."
    IMAGE_COUNT = 4  # Must be between 1 and 8
    # =======================================================

    print(f"--- 🖼️ Starting Direct Image Generation ---")

    generate_images(CLIENT, TARGET_TOPIC, IMAGE_COUNT, OUTPUT_DIR)