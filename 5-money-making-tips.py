import os
import io
import json
from google import genai
from google.genai import types
from google.cloud import texttospeech
from PIL import Image
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# NOTE: For video generation, you would need to install moviepy:
# from moviepy.editor import ImageClip, concatenate_videoclips

# --- Configuration ---
# Ensure your API key is set as an environment variable (GEMINI_API_KEY)
# If not, you can uncomment the line below and replace "YOUR_API_KEY"
os.environ["GEMINI_API_KEY"] = ""

try:
    CLIENT = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
    exit()

OUTPUT_IMAGE_DIR = "money_making_shorts_images"


# --- Step 1: Generate Money-Making Opportunities (Gemini) ---

def generate_opportunities(client: genai.Client) -> list:
    """Uses Gemini 2.5 Pro to generate a list of 5 money-making opportunities."""

    prompt = """
    Generate a list of exactly 5 distinct, creative, and actionable money-making opportunities 
    that could be suitable for a YouTube Short video. Each opportunity should be a short, compelling 
    title/concept.
    """

    # Define the desired JSON output structure for reliable parsing
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

def generate_images(client: genai.Client, opportunity_list: list) -> dict:
    """Uses the Imagen model to generate an image for each opportunity."""

    if not opportunity_list:
        return {}

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    image_paths = {}

    print(f"\n🎨 Step 2: Generating {len(opportunity_list)} images using Imagen...")

    # The visual style requested ("nono banana") is interpreted as vibrant, fun, and high-quality
    style_prompt_suffix = (
        "Vibrant, cartoonish 3D render, close-up shot, minimalist background, "
        "ultra-detailed, perfect for vertical video (9:16)."
    )

    for i, opportunity in enumerate(opportunity_list):
        full_prompt = f"The concept: '{opportunity}'. {style_prompt_suffix}"

        try:
            result = client.models.generate_images(
                model='imagen-4.0-fast-generate-001',
                prompt=full_prompt,
                config=dict(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="9:16",
                )
            )

            if result.generated_images:
                image_bytes = result.generated_images[0].image.image_bytes
                img = Image.open(io.BytesIO(image_bytes))

                file_path = os.path.join(OUTPUT_IMAGE_DIR, f"opportunity_{i + 1}.jpg")
                img.save(file_path)
                image_paths[opportunity] = file_path
                print(f"   ✅ Image {i + 1} saved for: '{opportunity}'")
            else:
                print(f"   ❌ Could not generate image for: {opportunity}")

        except Exception as e:
            print(f"   ❌ Error during image generation for '{opportunity}': {e}")

    return image_paths


# --- Step 3: Stitch Images into Video (Conceptual Placeholder) ---

def stitch_video(image_paths: dict, output_filename: str = "youtube_short.mp4"):
    """
    Stitches images into a video using the MoviePy library.

    Args:
        image_paths: A dictionary containing image file paths.
        output_filename: The name for the final video file.
    """
    if not image_paths:
        print("\n🎬 Step 3: Video stitching skipped. No images to stitch.")
        return

    print(f"\n🎬 Step 3: Stitching images into video: {output_filename}...")

    # Define the duration for each image slide (e.g., 3 seconds per image)
    clip_duration = 6
    image_files = list(image_paths.values())

    try:
        # 1. Create a video clip object for each image
        clips = []
        for img_file in image_files:
            clip = ImageClip(img_file, duration=clip_duration)
            clips.append(clip)

        # 2. Concatenate all individual clips into one final clip
        try:
            # 1. Concatenate image clips (as before)
            final_clip = concatenate_videoclips(clips)

            # 2. Load the generated audio
            if audio_file_path and os.path.exists(audio_file_path):
                audio_clip = AudioFileClip(audio_file_path)

                # Ensure the video clip and audio clip have the same duration
                final_clip = final_clip.set_audio(audio_clip)
                final_clip = final_clip.set_duration(audio_clip.duration)  # Set video duration to match audio length

        # 3. Write the final video file (codec must be compatible with MP4/YouTube)
        final_clip.write_videofile(
            output_filename,
            codec='libx264',  # Standard codec for MP4/YouTube
            audio_codec='aac',
            fps=24  # Frames per second
        )

        print(f"   ✅ Video successfully created at: {output_filename}")

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
            model='gemini-2.5-pro',
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
    """
    Uses Google Cloud Text-to-Speech to generate an audio file from the transcript.

    Args:
        transcript_text: The full script generated by Gemini.
        output_path: The directory path to save the MP3 file.
        output_filename: The name of the resulting audio file.

    Returns:
        The file path to the generated audio file.
    """
    if not transcript_text:
        print("   ❌ Audio generation skipped: No transcript text provided.")
        return ""

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=transcript_text)

    # Configure the voice (Choose a high-quality, engaging voice)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        # WaveNet voices are high-quality and sound very natural
        name="en-US-Wavenet-D",  # A good male voice
        # name="en-US-Standard-C", # A good female voice (less quality than WaveNet)
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    # Select the type of audio file (MP3 is widely compatible)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        # Increase the speed for a punchier, YouTube Short feel (1.1 = 10% faster)
        speaking_rate=1.1
    )

    print("\n🎤 Step 5: Generating voiceover using Google Cloud TTS...")

    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        final_file_path = os.path.join(output_path, output_filename)

        # The response's audio_content is binary.
        with open(final_file_path, "wb") as out:
            out.write(response.audio_content)

        print(f"   ✅ Audio content successfully saved to: {final_file_path}")
        return final_file_path

    except Exception as e:
        print(f"   ❌ An error occurred during TTS generation. Check ADC authentication: {e}")
        return ""

# --- Main Execution ---

if __name__ == "__main__":

    # 1. Generate opportunities
    opportunities = generate_opportunities(CLIENT)

    # 2. Generate images
    image_map = generate_images(CLIENT, opportunities)

    # 3. Stitch video (Conceptual Step)
    stitch_video(image_map)

    # 4. Generate final YouTube content
    youtube_final_content = generate_youtube_content(CLIENT, opportunities)

    # --- Final Output Summary ---
    if youtube_final_content:
        print("\n" + "=" * 50)
        print("         FINAL YOUTUBE CONTENT SUMMARY")
        print("=" * 50)
        print(f"TITLE: {youtube_final_content.get('title')}")
        print("\nDESCRIPTION:")
        print(youtube_final_content.get('description'))
        print(f"\nTAGS: {youtube_final_content.get('tags')}")
        print("\nTRANSCRIPT (Use for Voiceover/Captions):")
        print("----------------------------------------")
        print(youtube_final_content.get('transcript'))
        print("----------------------------------------")

    print("\n🎉 ALL AI CONTENT GENERATION COMPLETE! 🎉")
    print("The final step is running the conceptual 'stitch_video' function using MoviePy/FFmpeg.")