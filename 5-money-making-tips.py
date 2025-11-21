import os
import io
import json
import re
from google import genai
from google.genai import types
from google.cloud import texttospeech
from PIL import Image as PillowImage
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips


try:
    CLIENT = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
    exit()


# --- UTILITY FUNCTION: Create Numbered Output Directory ---
def get_next_output_directory(base_dir: str = "Generated_Shorts") -> str:
    """
    Creates and returns the path to the next sequentially numbered directory (e.g., Generated_Shorts/run_001).
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


# --- Step 1: Generate Money-Making Opportunities (Gemini) ---

def generate_opportunities(client: genai.Client) -> list:
    # ... (function body remains the same)

    # Using 'gemini-2.5-flash' for cost-efficiency on text generation
    # ... (rest of the function body)

    prompt = """
    Generate a list of exactly 5 distinct, creative, and actionable money-making opportunities 
    that could be suitable for a YouTube Short video. Each opportunity should be a short, compelling 
    title/concept.
    """

    opportunity_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "opportunities": types.Schema(
                type=types.Type.ARRAY,
                description="A list of the 5 money-making opportunity concepts.",
                items=types.Schema(type=types.Type.STRING)
            )
        },
        required=["opportunities"]
    )

    print("🤖 Step 1: Generating 5 money-making opportunities...")

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=opportunity_schema,
            ),
        )

        opportunities_data = json.loads(response.text)
        opportunity_list = opportunities_data.get("opportunities", [])

        print(f"   ✅ Generated {len(opportunity_list)} opportunities.")
        return opportunity_list

    except Exception as e:
        print(f"   ❌ Error during opportunity generation: {e}")
        return []


# --- Step 2: Generate Images (Imagen) ---

def generate_images(client: genai.Client, opportunity_list: list, output_path: str) -> dict:
    """Uses the Imagen model to generate an image for each opportunity and saves to output_path."""

    if not opportunity_list:
        return {}

    image_paths = {}

    print(f"\n🎨 Step 2: Generating {len(opportunity_list)} images using Imagen...")

    style_prompt_suffix = (
        "Hyper-detailed editorial photograph, captured in sharp 8K resolution."
        "Focus on dynamic contrast and dramatic rim lighting. Crisp textures, professional studio setting,"
        "Vertical screen composition."
    )

    for i, opportunity in enumerate(opportunity_list):
        full_prompt = f"The concept: '{opportunity}'. {style_prompt_suffix}"

        try:
            result = client.models.generate_images(
                model='imagen-4.0-fast-generate-001',  # Using fast model for quick generation
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

                file_path = os.path.join(output_path, f"opportunity_{i + 1}.jpg")
                img.save(file_path)
                image_paths[opportunity] = file_path
                print(f"   ✅ Image {i + 1} saved for: '{opportunity}'")
            else:
                print(f"   ❌ Could not generate image for: {opportunity}")

        except Exception as e:
            print(f"   ❌ Error during image generation for '{opportunity}': {e}")

    return image_paths


# --- Step 3: Stitch Images and Audio into Video ---

def stitch_video(image_paths: dict, output_path: str, audio_file_path: str = None,
                 output_filename: str = "youtube_short.mp4"):
    """
    Stitches images and an optional audio file into a video using the MoviePy library.
    """
    if not image_paths:
        print("\n🎬 Step 3: Video stitching skipped. No images to stitch.")
        return

    final_output_path = os.path.join(output_path, output_filename)
    print(f"\n🎬 Step 3: Stitching assets into video: {final_output_path}...")

    # Define the duration for each image slide (6 seconds in your code)
    # NOTE: This duration will be overridden if an audio file is provided.
    clip_duration = 6
    image_files = list(image_paths.values())

    try:
        # 1. Create a video clip object for each image
        clips = []
        for img_file in image_files:
            # We must set a duration, even if it's overridden later
            clip = ImageClip(img_file, duration=clip_duration)
            # Use vfx.resize for better compatibility
            # clip = clip.fx(vfx.resize, width=1080)
            clips.append(clip)

        # 2. Concatenate all individual clips into one final video clip
        final_clip = concatenate_videoclips(clips)

        # 3. Add Audio if available
        if audio_file_path and os.path.exists(audio_file_path):
            audio_clip = AudioFileClip(audio_file_path)

            # Set the video duration to match the audio clip length
            final_clip = final_clip.set_duration(audio_clip.duration)
            final_clip = final_clip.set_audio(audio_clip)

        # 4. Write the final video file
        final_clip.write_videofile(
            final_output_path,
            codec='libx264',
            audio_codec='aac',
            fps=24,  # Frames per second
            logger=None
        )

        print(f"   ✅ Video successfully created at: {final_output_path}")

    except ImportError:
        print("   ❌ ERROR: MoviePy or its dependencies (like FFmpeg) are not installed or accessible.")
        print("   Please run 'pip install moviepy' and ensure FFmpeg is set up correctly.")
    except Exception as e:
        print(f"   ❌ An error occurred during video generation: {e}")


# --- Step 4: Generate YouTube Content and Transcript (Gemini) ---

def generate_youtube_content(client: genai.Client, opportunity_list: list) -> dict:
    """Generates the YouTube Short title, description, tags, and a video transcript."""

    if not opportunity_list:
        return {}

    opportunities_str = "\n".join([f"- {opp}" for opp in opportunity_list])

    prompt = f"""
    You are a professional content creator crafting a YouTube Short video about money-making opportunities.
    The video features these 5 opportunities:
    {opportunities_str}

    Based on these, generate the following in English, formatted as a JSON object:
    1.  **A catchy YouTube Short title** (under 100 characters).
    2.  **A concise YouTube Short description** (under 500 characters), including a strong call to action and relevant hashtags.
    3.  **A comma-separated list of relevant YouTube tags** (e.g., "makemoney, sidehustle, easycash").
    4.  **A detailed video transcript script** for the entire 5-opportunity video (under 500 characters). The script must be energetic and clear. For each opportunity, provide 1-2 sentences of introduction and 2-3 sentences of explanation. The script should start with an engaging hook and end with a call to action.
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

    print("\n📝 Step 4: Generating YouTube Short content...")

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
        print("   ✅ YouTube Content Generated.")
        return youtube_data

    except Exception as e:
        print(f"   ❌ Error during YouTube content generation: {e}")
        return {}


# --- Step 5: Generate Audio from Transcript ---

def generate_audio(transcript_text: str, output_path: str, output_filename: str = "voiceover.mp3") -> str:
    # ... (function body remains the same) ...
    """
    Uses Google Cloud Text-to-Speech to generate an audio file from the transcript.
    """
    if not transcript_text:
        print("   ❌ Audio generation skipped: No transcript text provided.")
        return ""

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=transcript_text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.1
    )

    print("\n🎤 Step 5: Generating voiceover using Google Cloud TTS...")

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
        print(f"   ❌ An error occurred during TTS generation. Check ADC authentication: {e}")
        return ""


# --- Main Execution ---

if __name__ == "__main__":

    # --- 1. Setup Environment and Output Directory ---
    output_dir = get_next_output_directory()

    # --- 2. Text Generation: Get 5 money-making opportunities ---
    opportunities = generate_opportunities(CLIENT)
    if not opportunities:
        print("\n❌ Pipeline stopped: Failed to generate opportunities.")
        exit()

    # --- 3. Image Generation: Create a visual for each opportunity ---
    image_map = generate_images(CLIENT, opportunities, output_dir)
    if len(image_map) != len(opportunities):
        print("\n⚠️ Warning: Did not generate all images. Proceeding with available images.")

    # --- 4. YouTube Content Generation & Metadata Save ---
    youtube_content = generate_youtube_content(CLIENT, opportunities)
    transcript = youtube_content.get('transcript', '')

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(youtube_content, f, indent=4)
    print(f"📝 YouTube metadata and transcript saved to: {metadata_file}")

    # --- 5. Audio Generation: Create the voiceover from the transcript ---
    audio_file_path = generate_audio(transcript, output_dir)

    # --- 6. Video Stitching: Combine images and audio into the final MP4 video ---
    if image_map:
        stitch_video(image_map, output_dir, audio_file_path=audio_file_path)
    else:
        print("\n❌ Final video creation skipped: No images were available.")

    print("\n\n🎉 AI AUTOMATION PIPELINE COMPLETE! All assets are in the output directory. 🎉")
