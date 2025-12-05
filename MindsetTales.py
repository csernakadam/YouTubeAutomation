import os
import io
import json
import re
from google import genai
from google.genai import types
from google.cloud import texttospeech
from PIL import Image as PillowImage


# --- 1. CONFIGURATION CLASS ---

class Config:
    """
    Configuration settings for the MindsetTales storyboard generation pipeline.
    """
    # ----------------------------------------------------
    # --- STORY & LANGUAGE SETTINGS ---
    # ----------------------------------------------------
    # Topic for the mindset story featuring the beggar and businessman.
    CARTOON_TOPIC = "How the Beggar Became a Businessman: The Mindset of Exchange."

    # Target language set to English
    TARGET_LANGUAGE = "English"

    # Total number of images/scenes to generate.
    IMAGE_COUNT = 9

    # Total word count for the transcript.
    TRANSCRIPT_TOTAL_WORDS = 400

    # Transcript content structure percentages (MUST sum to 1.0 or 100%).
    CONTENT_RATIOS = {
        "HOOK": 0.10,  # 10%
        "SHORT_CTA": 0.05,  # 5%
        "STORY": 0.70,  # 70%
        "LONG_CTA": 0.15  # 15%
    }

    # ----------------------------------------------------
    # --- VISUAL STYLE SETTINGS (Locked) ---
    # ----------------------------------------------------
    # New parameter for image generation consistency (Seed).
    IMAGE_SEED = 4444

    VISUAL_STYLE_PROMPT = (
        "2D Flat Design Style, ultra-detailed, simple shapes, thick black outlines (2px), zero realism. "
        "Background: Soft pastel gradient, light blue to peach orange, blurred depth-of-field, subtle lens flare, gentle vignette. "
        "Setting: Crowded railway station platform with repeating assets like blooming flowers and spinning golden coins. "
        "Character Style: Chibi-proportion, head is 40% of body height. HUGE eyes (30% of face), solid colors, simple shapes. "
        "Emotions: Achieved by large shiny highlights in the eyes and clear mouth shapes. Sadness = gray/blue watery eyes. Realization = bright emerald eyes with white sparkles. "
        "Color Palette: Saturated characters (red flowers, deep blue suits) against soft pastel backgrounds. "
        "Lighting: Soft, permanent lens flare in a top corner, edges have a slight vignette. "
        "Action: Focus on high-emotion moments like holding out an oversized four-fingered hand (hope) or a determined fist. "
        "Quality: Ultra-high quality, cinematic, centered composition."
    )

    # ----------------------------------------------------
    # --- DIRECTORY SETTINGS ---
    # ----------------------------------------------------
    BASE_OUTPUT_DIR = "Generated_MindsetTales_Storyboards"

    if sum(CONTENT_RATIOS.values()) != 1.0:
        raise ValueError("The sum of CONTENT_RATIOS must equal 1.0 (100%). Please adjust the percentages.")


# Initialize the Gemini Client
try:
    CLIENT = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
    exit()


# --- UTILITY FUNCTION: Create Numbered Output Directory ---

def get_next_output_directory(base_dir: str) -> str:
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


# ----------------------------------------------------------------------------------------------------------------------

# --- Step 1: Generate Storyboard Descriptions (Gemini) ---

def generate_storyboard_descriptions(client: genai.Client, topic: str, count: int, target_language: str) -> list:
    """
    Generates a list of descriptions for each slide/scene of the MindsetTales storyboard.
    """

    prompt = f"""
    Generate exactly {count} distinct, sequential, and highly descriptive **scene descriptions**
    for a motivational and educational MindsetTales short story in {target_language} about the topic: **"{topic}"**.

    The story must feature a beggar and a businessman in a railway station setting.

    The descriptions must:
    1.  Clearly define the visual content for each of the {count} slides, focusing on key emotional moments (beggar's confusion, businessman's sharp smile, beggar's realization, happy exchange).
    2.  Together, form a complete, engaging, and educational narrative arc.
    3.  Focus the descriptions on the main STORY portion of the video.

    The final list of {count} descriptions MUST be structured as a JSON array of strings.
    Example: ["Description 1", "Description 2", ...]
    """

    description_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "scenes": types.Schema(
                type=types.Type.ARRAY,
                description=f"A list of {count} scene descriptions for the MindsetTales story.",
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


# ----------------------------------------------------------------------------------------------------------------------

# --- Step 2: Generate Images (Imagen) ---

def generate_images(client: genai.Client, scene_list: list, output_path: str, style_prompt: str,
                    image_seed: int) -> dict:
    """Uses the Imagen model to generate an image for each scene, locking the style with a seed."""

    if not scene_list:
        return {}

    image_paths = {}

    print(f"\n🎨 Step 2: Generating {len(scene_list)} images with custom style (Seed: {image_seed})...")

    for i, scene_description in enumerate(scene_list):

        # Translation is skipped as the target language is English
        english_scene = scene_description

        # Combine scene-specific description with the fixed style prompt
        full_prompt = f"{english_scene}. {style_prompt}"

        try:
            result = client.models.generate_images(
                model='imagen-4.0-generate-001',
                prompt=full_prompt,
                config=dict(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="16:9",
                    seed=image_seed,
                    add_watermark=False,
                    image_size="2K"
                )
            )

            if result.generated_images:
                image_bytes = result.generated_images[0].image.image_bytes
                img = PillowImage.open(io.BytesIO(image_bytes))

                file_path = os.path.join(output_path, f"scene_{i + 1}.jpg")
                img.save(file_path)
                image_paths[scene_description] = file_path
                print(f"   ✅ Image {i + 1} saved for scene: '{scene_description}'")
            else:
                print(f"   ❌ Could not generate image for: {scene_description}")

        except Exception as e:
            print(f"   ❌ Error during image generation for '{scene_description}': {e}")

    return image_paths


# ----------------------------------------------------------------------------------------------------------------------

# --- Step 3: Generate YouTube Content and Transcript (Gemini) ---

def generate_youtube_content(client: genai.Client, scene_list: list, topic: str, target_language: str,
                             total_word_count: int, content_ratios: dict) -> dict:
    """Generates the structured content (title, description, tags) and the full SSML transcript."""

    if not scene_list:
        return {}

    scenes_str = "\n".join([f"- {i + 1}. {scene}" for i, scene in enumerate(scene_list)])

    # Calculate word counts based on ratios
    hook_wc = round(total_word_count * content_ratios["HOOK"])
    short_cta_wc = round(total_word_count * content_ratios["SHORT_CTA"])
    story_wc = round(total_word_count * content_ratios["STORY"])
    long_cta_wc = round(total_word_count * content_ratios["LONG_CTA"])
    final_wc = hook_wc + short_cta_wc + story_wc + long_cta_wc

    prompt = f"""
    You are a professional content writer creating a motivational, educational, {target_language} MindsetTales short script 
    about the topic: **"{topic}"**. The video will use {len(scene_list)} sequential scene descriptions for the main story part.

    INSTRUCTION: The final **TRANSCRIPT MUST contain ONLY the continuous spoken story**.
    **DO NOT INCLUDE** any speaker labels, character names, or directional cues (like NARRATOR: or [sound effect]) within the <speak> tags.

    The script must adhere to the following strict structure and word counts:
    1.  **HOOK (approx. {hook_wc} words):** A very engaging opening, setting up the problem.
    2.  **SHORT CALL TO ACTION (approx. {short_cta_wc} words):** A quick instruction (e.g., "Like and subscribe!").
    3.  **STORY (approx. {story_wc} words):** The main narrative, which must be perfectly aligned with the {len(scene_list)} Scene Descriptions below.
    4.  **LONG CALL TO ACTION (approx. {long_cta_wc} words):** The moral of the story and a final closing message/pitch reflecting the MindsetTales channel ethos: "Change your story. Change your life."

    Total Target Word Count: **{final_wc} words**.

    Scene Descriptions (for the STORY segment):
    {scenes_str}

    Based on the above, generate the required output as a JSON object:

    1.  **A catchy YouTube video title** in {target_language} (under 100 characters).
    2.  **A concise YouTube video description** in {target_language}, including a strong call to action and relevant hashtags (~ 700 characters).
    3.  **A comma-separated list of relevant YouTube tags** in {target_language} (under 500 characters).
    4.  **A detailed video transcript script** in {target_language}. The script must strictly adhere to these SSML rules:
        * The entire transcript MUST be wrapped in **<speak></speak> tags**.
        * The script must be split into segments reflecting the four structural parts.
        * Use **<break strength="strong"/>** tags to create distinct pauses: **one must occur between each of the four main structural parts.**
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
        print("   ✅ Content Generated and Structured.")
        return youtube_data

    except Exception as e:
        print(f"   ❌ Error during content generation: {e}")
        return {}


# ----------------------------------------------------------------------------------------------------------------------

# --- Step 4: Generate Audio from Transcript ---

def generate_audio(transcript_text: str, output_path: str, target_language: str,
                   output_filename: str = "voiceover.mp3") -> str:
    """
    Uses Google Cloud Text-to-Speech to generate an audio file from the SSML transcript.
    """
    if not transcript_text:
        print("   ❌ Audio generation skipped: No transcript text provided.")
        return ""

    # TTS VOICE CONFIGURATION
    LANGUAGE_CONFIG = {
        "English": {"code": "en-US", "name": "en-US-Studio-Q"},  # High-quality English voice
        "Hungarian": {"code": "hu-HU", "name": "hu-HU-Chirp3-HD-Achernar"},
        "Spanish": {"code": "es-ES", "name": "es-ES-Wavenet-C"},
    }

    config = LANGUAGE_CONFIG.get(target_language, LANGUAGE_CONFIG["English"])

    try:
        client = texttospeech.TextToSpeechClient()
    except Exception as e:
        print(f"   ❌ TTS Client Error: {e}. Check Google Cloud ADC authentication.")
        return ""

    synthesis_input = texttospeech.SynthesisInput(ssml=transcript_text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=config["code"],
        name=config["name"],
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0
    )

    print(f"\n🎤 Step 4: Generating {target_language} voiceover...")

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


# ----------------------------------------------------------------------------------------------------------------------

# --- Main Execution ---

if __name__ == "__main__":

    # Validate and print configuration
    print("--- ⚙️ MINDSETTALES PIPELINE CONFIGURATION ---")
    print(f"Topic: {Config.CARTOON_TOPIC}")
    print(f"Language: {Config.TARGET_LANGUAGE}")
    print(f"Scenes: {Config.IMAGE_COUNT}")
    print(f"Total Words: {Config.TRANSCRIPT_TOTAL_WORDS}")
    print(f"Image Seed: {Config.IMAGE_SEED} (Locked)")
    print(f"Structure: {Config.CONTENT_RATIOS}")
    print(f"Style: {Config.VISUAL_STYLE_PROMPT[:50]}...")
    print("--------------------------------------------\n")

    # 1. Setup Environment and Output Directory
    output_dir = get_next_output_directory(Config.BASE_OUTPUT_DIR)

    # 2. Storyboard Generation: Get scene descriptions
    scenes = generate_storyboard_descriptions(
        CLIENT,
        Config.CARTOON_TOPIC,
        Config.IMAGE_COUNT,
        Config.TARGET_LANGUAGE
    )
    if not scenes:
        print("\n❌ Pipeline stopped: Failed to generate scene descriptions.")
        exit()

    # 3. Image Generation: Create visuals
    image_map = generate_images(
        CLIENT,
        scenes,
        output_dir,
        Config.VISUAL_STYLE_PROMPT,
        Config.IMAGE_SEED  # <-- Passing the locked seed
    )
    if len(image_map) != len(scenes):
        print("\n⚠️ Warning: Did not generate all images. Proceeding with available assets.")

    # 4. YouTube Content Generation & Metadata Save
    youtube_content = generate_youtube_content(
        CLIENT,
        scenes,
        Config.CARTOON_TOPIC,
        Config.TARGET_LANGUAGE,
        Config.TRANSCRIPT_TOTAL_WORDS,
        Config.CONTENT_RATIOS
    )
    transcript = youtube_content.get('transcript', '')

    metadata_file = os.path.join(output_dir, f"metadata_{Config.TARGET_LANGUAGE}.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(youtube_content, f, indent=4, ensure_ascii=False)
    print(f"📝 YouTube metadata and transcript saved to: {metadata_file}")

    # 5. Audio Generation: Create the voiceover
    audio_file_path = generate_audio(transcript, output_dir, Config.TARGET_LANGUAGE)

    print("\n\n🎉 MINDSETTALES PIPELINE COMPLETE! 🎉")
    print(f"Final assets are located in: {output_dir}")