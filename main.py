from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch
from transformers import CLIPProcessor, CLIPModel
import tempfile
from pathlib import Path
import logging
import traceback
import whisper
from transformers import pipeline
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx.all import resize, crop
from moviepy.video.VideoClip import ColorClip
import cv2
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CLIP model
try:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    logger.info("CLIP model loaded successfully")
except Exception as e:
    logger.error(f"Error loading CLIP model: {str(e)}")
    logger.error(traceback.format_exc())

# Initialize Whisper model for speech recognition
whisper_model = whisper.load_model("base")

# Create necessary directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def detect_beats(audio_path):
    """Detect beats in the audio file."""
    try:
        logger.info(f"Loading audio file: {audio_path}")
        y, sr = librosa.load(audio_path)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        logger.info(f"Detected {len(beat_times)} beats with tempo {tempo}")
        return beat_times, tempo
    except Exception as e:
        logger.error(f"Error in detect_beats: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def analyze_clip_with_clip(clip, action_prompts):
    """Analyze a video clip using CLIP to determine if it contains specific actions."""
    try:
        # Extract a frame from the middle of the clip
        frame = clip.get_frame(clip.duration / 2)
        
        # Process the frame with CLIP
        inputs = processor(images=frame, text=action_prompts, return_tensors="pt", padding=True)
        
        # Get CLIP scores
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Return probabilities for each prompt
        return probs.detach().numpy().flatten()
    except Exception as e:
        logger.error(f"Error in analyze_clip_with_clip: {str(e)}")
        logger.error(traceback.format_exc())
        return np.zeros(len(action_prompts))

def extract_lyrics(audio_path):
    """Extract lyrics from the audio file using Whisper."""
    try:
        logger.info(f"Extracting lyrics from {audio_path}")
        result = whisper_model.transcribe(audio_path)
        lyrics = result["text"]
        logger.info(f"Extracted lyrics: {lyrics[:100]}...")  # Log first 100 chars
        return lyrics
    except Exception as e:
        logger.error(f"Error extracting lyrics: {str(e)}")
        logger.error(traceback.format_exc())
        return ""

def generate_prompts_from_lyrics(lyrics):
    """Generate visual prompts based on lyrics."""
    try:
        # Split lyrics into phrases
        phrases = [p.strip() for p in lyrics.split('.') if p.strip()]
        
        # Generate visual interpretations of each phrase
        visual_prompts = []
        for phrase in phrases:
            # Basic prompt generation (can be enhanced with more sophisticated NLP)
            words = phrase.lower().split()
            if len(words) > 2:  # Only use meaningful phrases
                # Create a visual interpretation of the phrase
                visual_prompt = f"a football player {phrase}"
                visual_prompts.append(visual_prompt)
        
        # Add some generic football prompts
        visual_prompts.extend([
            "a close-up shot of a football player",
            "a football player in focus with blurred background",
            "a tight shot of a football player's face",
            "a football player filling most of the frame",
            "a football player in a close-up action shot"
        ])
        
        logger.info(f"Generated {len(visual_prompts)} visual prompts from lyrics")
        return visual_prompts
    except Exception as e:
        logger.error(f"Error generating prompts: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def find_subject_center(frame, processor, model):
    """Find the center of the subject (player) in the frame using CLIP attention."""
    # Convert frame to PIL Image if it's numpy array
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    
    # Get CLIP attention map for "football player"
    inputs = processor(images=frame, text=["a football player"], return_tensors="pt")
    outputs = model(**inputs)
    
    # Get attention weights
    attention = outputs.vision_model.pooler_output.detach().numpy()
    
    # Convert attention to heatmap
    heatmap = cv2.resize(attention[0], (frame.size[0], frame.size[1]))
    
    # Find the center of mass of the attention
    y_coords, x_coords = np.where(heatmap > np.mean(heatmap))
    if len(x_coords) > 0 and len(y_coords) > 0:
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
    else:
        # If no clear subject is found, use the center of the frame
        center_x = frame.size[0] // 2
        center_y = frame.size[1] // 2
    
    return center_x, center_y

def convert_to_portrait(clip, processor=processor, model=model):
    """Convert a video clip to portrait mode (9:16 aspect ratio) with smart centering"""
    target_aspect_ratio = 9/16  # Portrait mode aspect ratio
    
    def transform_frame(get_frame, t):
        # Get the current frame
        frame = get_frame(t)
        
        # Find subject center in the current frame
        center_x, _ = find_subject_center(frame, processor, model)
        
        # Get original dimensions
        h, w = frame.shape[:2]
        current_aspect_ratio = w/h
        
        if current_aspect_ratio > target_aspect_ratio:
            # Video is too wide, need to crop width
            new_w = int(h * target_aspect_ratio)
            
            # Calculate crop boundaries based on subject center
            x1 = max(0, min(center_x - new_w//2, w - new_w))
            x2 = x1 + new_w
            
            # Crop the frame
            cropped_frame = frame[:, x1:x2]
            return cropped_frame
        else:
            # Handle vertical videos similar to before
            new_h = int(w / target_aspect_ratio)
            if new_h < h:
                # Need to crop height
                y_center = h/2
                y1 = int(y_center - new_h/2)
                cropped_frame = frame[y1:y1+new_h, :]
                return cropped_frame
            else:
                # Add black bars for vertical videos
                final_h = int(w / target_aspect_ratio)
                result = np.zeros((final_h, w, 3), dtype='uint8')
                y_offset = (final_h - h) // 2
                result[y_offset:y_offset+h, :] = frame
                return result
    
    # Create a new clip with the transform applied to each frame
    new_clip = clip.fl(transform_frame)
    return new_clip

def analyze_video_clips(video_path, clip_duration=2.0, action_prompts=None, lyrics=None):
    try:
        logger.info(f"Loading video file: {video_path}")
        video = VideoFileClip(video_path)
        duration = video.duration
        clips = []
        clip_scores = []
        
        # Updated default action prompts focused on close-up football plays
        if action_prompts is None:
            action_prompts = [
                # Positive prompts (specific to football highlights)
                "a close-up of a football player catching a ball",
                "a football player making an athletic catch",
                "a tight shot of a football player running with the ball",
                "a close-up of a football player breaking tackles",
                "a football player making a spectacular play",
                "a football receiver running a route",
                "a close-up of a football player celebrating",
                
                # Negative prompts (what we want to avoid)
                "a wide shot of a football field",
                "a football game with scoreboard visible",
                "crowd watching football",
                "football stadium aerial view",
                "football sideline view"
            ]
        
        # Increase weight for positive clips and stronger penalty for wide shots
        for start_time in np.arange(0, duration, clip_duration):
            end_time = min(start_time + clip_duration, duration)
            if end_time - start_time > 0.5:
                clip = video.subclip(start_time, end_time)
                scores = analyze_clip_with_clip(clip, action_prompts)
                
                # Split scores and adjust weights
                positive_scores = scores[:7]  # First 7 prompts are positive
                negative_scores = scores[7:]  # Last 5 prompts are negative
                
                # Increased weight for positive scores
                positive_mean = np.mean(positive_scores) * 1.5
                negative_mean = np.mean(negative_scores)
                
                # Stronger penalty for wide shots
                final_score = positive_mean - (negative_mean * 0.8)
                
                clips.append(clip)
                clip_scores.append(final_score)

        return clips, clip_scores
    except Exception as e:
        logger.error(f"Error in analyze_video_clips: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_edit(video_path, audio_path, output_path, target_duration=60, portrait=False):
    try:
        logger.info(f"Starting video edit creation with target duration: {target_duration} seconds")
        
        # Get beat times first
        beat_times, tempo = detect_beats(audio_path)
        
        # Adjust clip duration to be shorter for more dynamic editing
        base_clip_duration = 1.5  # Reduced from 2.0 for more dynamic cuts
        tempo_factor = 120 / tempo
        clip_duration = base_clip_duration * tempo_factor
        
        # Analyze video and get clips
        video_clips, clip_scores = analyze_video_clips(video_path, clip_duration)
        
        if not video_clips:
            raise ValueError("No valid clips could be extracted from the video")
        
        # Sort clips by score
        sorted_indices = np.argsort(clip_scores)[::-1]
        sorted_clips = [video_clips[i] for i in sorted_indices]
        
        # Select clips and align them with beats
        selected_clips = []
        current_time = 0
        beat_index = 0
        
        while current_time < target_duration and beat_index < len(beat_times):
            # Find next beat time
            next_beat = beat_times[beat_index]
            
            # Find a suitable clip
            for i, clip in enumerate(sorted_clips):
                if clip.duration <= (target_duration - current_time):
                    clip = clip.without_audio()
                    selected_clips.append(clip)
                    current_time += clip.duration
                    sorted_clips.pop(i)
                    break
            
            beat_index += 1
        
        # Calculate how many clips we need to reach target duration
        current_duration = 0
        used_indices = set()
        
        # Second pass: select the best clips until we reach target duration
        for i in range(len(selected_clips)):
            if current_duration >= target_duration:
                break
            if i not in used_indices:
                clip = selected_clips[i]
                selected_clips[i] = clip.without_audio()
                used_indices.add(i)
                current_duration += clip.duration
        
        # If we haven't reached target duration, add more clips
        if current_duration < target_duration:
            remaining_duration = target_duration - current_duration
            # Try to find clips that fit the remaining duration
            for i in range(len(selected_clips)):
                if i not in used_indices:
                    clip = selected_clips[i]
                    if clip.duration <= remaining_duration:
                        selected_clips[i] = clip.without_audio()
                        used_indices.add(i)
                        current_duration += clip.duration
                        remaining_duration = target_duration - current_duration
                        if remaining_duration <= 0:
                            break
        
        logger.info(f"Selected {len(selected_clips)} clips with total duration: {current_duration:.2f} seconds")
        
        # Concatenate clips
        final_video = concatenate_videoclips(selected_clips)
        
        if portrait:
            final_video = convert_to_portrait(final_video)
        
        # Load and trim audio using AudioFileClip instead of VideoFileClip
        audio = AudioFileClip(audio_path)
        if audio.duration > final_video.duration:
            audio = audio.subclip(0, final_video.duration)
        final_video = final_video.set_audio(audio)
        
        # Write the final video
        logger.info(f"Writing final video to {output_path}")
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        logger.info("Video creation completed successfully")
    except Exception as e:
        logger.error(f"Error in create_edit: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/create-edit")
async def create_video_edit(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    duration: int = Form(60),  # Default to 60 seconds
    portrait_mode: bool = Form(False)  # New parameter for portrait mode
):
    try:
        logger.info(f"Received video and audio files with target duration: {duration} seconds")
        # Save uploaded files
        video_path = UPLOAD_DIR / video.filename
        audio_path = UPLOAD_DIR / audio.filename
        output_path = UPLOAD_DIR / "final_edit.mp4"
        
        logger.info(f"Saving video to {video_path}")
        with open(video_path, "wb") as video_file:
            video_file.write(await video.read())
        
        logger.info(f"Saving audio to {audio_path}")
        with open(audio_path, "wb") as audio_file:
            audio_file.write(await audio.read())
        
        # Create the edit with portrait mode option
        create_edit(str(video_path), str(audio_path), str(output_path), duration, portrait_mode)
        
        # Clean up input files
        os.remove(video_path)
        os.remove(audio_path)
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="final_edit.mp4"
        )
    
    except Exception as e:
        logger.error(f"Error in create_video_edit endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 