# AI Video Editor

An AI-powered video editing application that automatically creates synchronized edits by matching video clips to music beats.

## Features

- Upload any video (e.g., sports highlights, gameplay footage)
- Add background music
- Automatic beat detection and synchronization
- AI-powered clip selection
- Download the final edited video

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```bash
   python main.py
   ```
2. Open your web browser and navigate to `http://localhost:8000`
3. Upload your video and music files
4. Click "Create Edit" and wait for the processing to complete
5. Preview and download your edited video

## Technical Details

The application uses:
- FastAPI for the backend server
- MoviePy for video processing
- Librosa for audio beat detection
- CLIP for video understanding
- Modern HTML/CSS/JavaScript for the frontend

## Requirements

- Python 3.8+
- FFmpeg (required by MoviePy)
- Sufficient disk space for video processing
- Modern web browser

## License

MIT License 