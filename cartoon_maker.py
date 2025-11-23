import os
import io
import json
import re
from google import genai
from google.genai import types
from google.cloud import texttospeech
from PIL import Image as PillowImage


try:
    CLIENT = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
    exit()

# --- CONFIGURATION CONSTANTS ---
# Topic for the cartoon. This is the easily modifiable part.
CARTOON_TOPIC = "Hogyan kezeli Kisautó a nagy frusztrációt, ha elakad a sárban"
IMAGE_COUNT = 6
TRANSCRIPT_DURATION_SECONDS = 60  # Target duration for a ~1-minute short
# --- Set the Target Language here ---
TARGET_LANGUAGE = "Hungarian"
# ------------------------------------
# Stílus: 3D render, Claymation hatás, konzisztens főszereplővel (a kisautó).
DISNEY_STYLE_PROMPT = (
    "3D render, Claymation style, bright volumetric lighting, smooth texture, hyper-detailed, "
    "pastel colors, character is an adorable **small, round, bright blue car with huge friendly headlights and a smiling grille** 3D mascot, "
    "cinematic depth of field, **extremely high quality**, vertical composition."
)

# --- UTILITY FUNCTION: Create Numbered Output Directory ---
def get_next_output_directory(base_dir: str = "Generated_Storyboards") -> str:  # Changed base directory name
    """
    Creates and returns the path to the next sequentially numbered directory.
    """
    os.makedirs(base_dir, exist_ok=True)

    existing_dirs = os.listdir(base_dir)
    max_num = 0
    pattern = re.compile(r"^run_(\d{3})$")

    for name in existing_dirs:
        match = pattern.match(name)
        if match:
            max_num = max(max_num, int(match.group(1)))

    next_num = max_num + 1
    new_dir_name = f"run_{next_num:03d}"
    new_dir_path = os.path.join(base_dir, new_dir_name)

    os.makedirs(new_dir_path, exist_ok=True)
    print(f"📁 Created new output directory: {new_dir_path}")

    return new_dir_path


# --- Step 1: Generate Storyboard Descriptions (Gemini) ---

def generate_storyboard_descriptions(client: genai.Client, topic: str, count: int) -> list:
    """
    Generates a list of descriptions for each slide/scene of the cartoon storyboard.
    """

    prompt = f"""
    Generate exactly {count} distinct, sequential, and highly descriptive **scene descriptions**
    for a funny and educational cartoon short in {TARGET_LANGUAGE} about the topic: **"{topic}"**.

    The descriptions must:
    1.  Clearly define the visual content for each of the {count} slides.
    2.  Together, form a complete, funny, and educational narrative arc suitable for a short video.

    The final list of {count} descriptions MUST be structured as a JSON array of strings.
    Example: ["Description 1", "Description 2", ...]
    """

    description_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "scenes": types.Schema(
                type=types.Type.ARRAY,
                description=f"A list of {count} scene descriptions for the cartoon.",
                items=types.Schema(type=types.Type.STRING)
            )
        },
        required=["scenes"]
    )

    print(f"🤖 Step 1: Generating {count} scene descriptions for the topic: '{topic}'...")

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=description_schema,
            ),
        )

        descriptions_data = json.loads(response.text)
        description_list = descriptions_data.get("scenes", [])

        print(f"   ✅ Generated {len(description_list)} scene descriptions.")
        return description_list

    except Exception as e:
        print(f"   ❌ Error during description generation: {e}")
        return []


# --- Step 2: Generate Images (Imagen) ---

def generate_images(client: genai.Client, scene_list: list, output_path: str) -> dict:
    """Uses the Imagen model to generate an image for each scene and saves to output_path."""

    if not scene_list:
        return {}

    image_paths = {}

    print(f"\n🎨 Step 2: Generating {len(scene_list)} images in Disney style...")

    style_prompt = DISNEY_STYLE_PROMPT

    for i, scene_description in enumerate(scene_list):
        # Translate the description for the Imagen model (which performs better with English prompts)
        # We will use Gemini to translate the Hungarian description to English for the image prompt.
        try:
            translation_prompt = f"Translate the following scene description from Hungarian to English for an image generator: '{scene_description}'"
            translation_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=translation_prompt
            )
            english_scene = translation_response.text.strip()
        except Exception as e:
            print(f"   ❌ Error translating scene {i + 1}. Using original text. Error: {e}")
            english_scene = scene_description  # Fallback

        full_prompt = f"{english_scene}. {style_prompt}"

        try:
            result = client.models.generate_images(
                model='imagen-4.0-generate-001',
                prompt=full_prompt,
                config=dict(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="9:16",
                )
            )

            if result.generated_images:
                image_bytes = result.generated_images[0].image.image_bytes
                img = PillowImage.open(io.BytesIO(image_bytes))

                file_path = os.path.join(output_path, f"scene_{i + 1}.jpg")
                img.save(file_path)
                image_paths[scene_description] = file_path  # Store Hungarian description with path
                print(f"   ✅ Image {i + 1} saved for scene: '{scene_description}'")
            else:
                print(f"   ❌ Could not generate image for: {scene_description}")

        except Exception as e:
            print(f"   ❌ Error during image generation for '{scene_description}': {e}")

    return image_paths


# --- Step 3: Generate YouTube Content and Transcript (Gemini) ---

def generate_youtube_content(client: genai.Client, scene_list: list, topic: str, target_language: str,
                             duration_seconds: int) -> dict:
    """Generates the content (title, description, tags) and the full Hungarian transcript."""

    if not scene_list:
        return {}

    scenes_str = "\n".join([f"- {i + 1}. {scene}" for i, scene in enumerate(scene_list)])

    # Estimate the word count needed for the target language (Hungarian is less concise than English)
    # A standard speaking rate is 150 words/minute in English. Let's aim for a similar duration in Hungarian SSML.
    # The SSML rules are adjusted to ensure the script matches the new scene count.

    prompt = f"""
    You are a professional content writer creating a funny, educational, {target_language} cartoon short script 
    about the topic: **"{topic}"**. The video will use {len(scene_list)} sequential scenes.

    Based on the following {len(scene_list)} scene descriptions, generate the required output as a JSON object:

    Scene Descriptions:
    {scenes_str}

    1.  **A catchy YouTube Short title** in {target_language} (under 100 characters).
    2.  **A concise YouTube Short description** in {target_language} (under 500 characters), including a strong call to action and relevant hashtags.
    3.  **A comma-separated list of relevant YouTube tags** in {target_language}.
    4.  **A detailed video transcript script** in {target_language}. The script must be funny, clear, and designed to last approximately **{duration_seconds} seconds** when spoken. It must strictly adhere to these SSML rules:
        * The entire transcript MUST be wrapped in **<speak></speak> tags**.
        * The script must be split into {len(scene_list)} main segments, one for each scene, ensuring smooth transitions.
        * Use **<break strength="strong"/>** tags to create a distinct, segment-level pause: **one must occur between each of the {len(scene_list)} segments.**
        * Ensure the script starts with an engaging hook and ends with a clear, funny lesson or call to action.
    """

    youtube_content_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "title": types.Schema(type=types.Type.STRING),
            "description": types.Schema(type=types.Type.STRING),
            "tags": types.Schema(type=types.Type.STRING),
            "transcript": types.Schema(type=types.Type.STRING)
        },
        required=["title", "description", "tags", "transcript"]
    )

    print(f"\n📝 Step 3: Generating YouTube Content in {target_language}...")

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=youtube_content_schema,
            ),
        )

        youtube_data = json.loads(response.text)
        print("   ✅ Hungarian Content Generated.")
        return youtube_data

    except Exception as e:
        print(f"   ❌ Error during content generation: {e}")
        return {}

# --- Step 4: Generate Audio from Transcript ---

def generate_audio(transcript_text: str, output_path: str, target_language: str,
                   output_filename: str = "voiceover.mp3") -> str:
    """
    Uses Google Cloud Text-to-Speech to generate an audio file from the SSML transcript, selecting the voice based on target_language.
    """
    if not transcript_text:
        print("   ❌ Audio generation skipped: No transcript text provided.")
        return ""

    # Map target language to a suitable TTS voice and language code
    # NOTE: hu-HU-Wavenet-F is used for better quality (less robotic).
    LANGUAGE_CONFIG = {
        "Hungarian": {"code": "hu-HU", "name": "hu-HU-Chirp3-HD-Achernar"},
        "English": {"code": "en-US", "name": "en-US-Studio-Q"},
        "Spanish": {"code": "es-ES", "name": "es-ES-Wavenet-C"},
    }

    if target_language not in LANGUAGE_CONFIG:
        print(f"   ❌ ERROR: TTS configuration missing for language: {target_language}. Using English default.")
        config = LANGUAGE_CONFIG["English"]
    else:
        config = LANGUAGE_CONFIG[target_language]

    try:
        client = texttospeech.TextToSpeechClient()
    except Exception as e:
        print(f"   ❌ TTS Client Error: {e}. Check Google Cloud ADC authentication.")
        return ""

    synthesis_input = texttospeech.SynthesisInput(ssml=transcript_text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=config["code"],
        name=config["name"],
        ssml_gender=config["gender"]
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.
    )

    print(f"\n🎤 Step 4: Generating {target_language} voiceover using Google Cloud TTS...")

    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        final_file_path = os.path.join(output_path, output_filename)

        with open(final_file_path, "wb") as out:
            out.write(response.audio_content)

        print(f"   ✅ Audio content successfully saved to: {final_file_path}")
        return final_file_path

    except Exception as e:
        print(f"   ❌ An error occurred during TTS generation: {e}")
        return ""


# --- Main Execution ---

if __name__ == "__main__":

    # --- 1. Setup Environment and Output Directory ---
    output_dir = get_next_output_directory()

    # --- 2. Storyboard Generation: Get 6 scene descriptions (in Hungarian) ---
    scenes = generate_storyboard_descriptions(CLIENT, CARTOON_TOPIC, IMAGE_COUNT)
    if not scenes:
        print("\n❌ Pipeline stopped: Failed to generate scene descriptions.")
        exit()

    # --- 3. Image Generation: Create a Disney-style visual for each scene ---
    image_map = generate_images(CLIENT, scenes, output_dir)
    if len(image_map) != len(scenes):
        print("\n⚠️ Warning: Did not generate all images. Proceeding with available images.")

    # --- 4. YouTube Content Generation & Metadata Save (in Hungarian) ---
    youtube_content = generate_youtube_content(
        CLIENT, scenes, CARTOON_TOPIC, TARGET_LANGUAGE, TRANSCRIPT_DURATION_SECONDS
    )
    transcript = youtube_content.get('transcript', '')

    metadata_file = os.path.join(output_dir, "metadata_hu.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(youtube_content, f, indent=4,
                  ensure_ascii=False)  # 'ensure_ascii=False' for correct Hungarian character display
    print(f"📝 YouTube metadata and Hungarian transcript saved to: {metadata_file}")

    # --- 5. Audio Generation: Create the Hungarian voiceover from the SSML transcript ---
    audio_file_path = generate_audio(transcript, output_dir, TARGET_LANGUAGE)

    print("\n\n🎉 HUNGARIAN CARTOON STORYBOARD PIPELINE COMPLETE! 🎉")
    print(f"Assets (6 Disney-style images, Hungarian transcript, Hungarian voiceover) are in: {output_dir}")