# 🎨 GenAI Text-to-Image Generator

Generate stunning, high-quality images from text descriptions using Stable Diffusion AI models.

## ✨ Features

- **Text-to-Image Generation**: Create images from natural language descriptions
- **Multiple Styles**: Photorealistic, Digital Art, Anime, Cyberpunk, and more
- **Advanced Controls**: Adjust steps, guidance scale, dimensions, and seed
- **Batch Generation**: Generate multiple images at once
- **Negative Prompts**: Specify what to avoid in generated images
- **Auto-Enhancement**: Automatically improve prompts with quality tags
- **Generation History**: Track and save all your generations
- **Download Images**: Save generated images in high quality

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 6GB+ VRAM)
- ~10GB disk space for models

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd genai-text2image

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Run

```bash
# Launch Streamlit app
streamlit run app.py

# The first run will download models (~5GB)
# This may take 10-20 minutes depending on your connection
```

## 💻 Usage

### Web Interface

1. Launch the app: `streamlit run app.py`
2. Enter your text prompt
3. Choose a style from the sidebar
4. Adjust settings (optional)
5. Click "Generate Images"
6. Download your creations!

### Programmatic Usage

```python
from app import StableDiffusionGenerator

# Initialize generator
generator = StableDiffusionGenerator()

# Generate images
images = generator.generate(
    prompt="a beautiful sunset over mountains",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    num_images=1,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42  # For reproducibility
)

# Save image
images[0].save("output.png")
```

## 🎯 Example Prompts

### Photorealistic
```
A professional photograph of a cat sitting on a wooden table, 
golden hour lighting, shallow depth of field, 8k uhd
```

### Fantasy Art
```
A majestic dragon perched on a mountain peak, 
epic fantasy art, dramatic lighting, highly detailed
```

### Cyberpunk
```
A futuristic city street at night, neon signs, 
rain-soaked pavement, cyberpunk aesthetic
```

### Portrait
```
Portrait of a wise old wizard with a long beard, 
magical aura, fantasy character design, detailed
```

## ⚙️ Parameters Guide

### Basic Parameters

- **Width/Height**: Image dimensions (multiples of 8)
  - 512x512: Standard, fast
  - 768x768: Higher quality, slower
  - 512x768: Portrait
  - 768x512: Landscape

- **Number of Images**: 1-4 images per generation
  - More images = longer generation time

### Advanced Parameters

- **Inference Steps**: (20-100)
  - 20-30: Fast, lower quality
  - 50: Balanced (recommended)
  - 75-100: Highest quality, slower

- **Guidance Scale**: (1-20)
  - 1-5: More creative/random
  - 7-8: Balanced (recommended)
  - 10-20: Strictly follows prompt

- **Seed**: 
  - -1: Random each time
  - Fixed number: Reproducible results

## 🎨 Styles

Available preset styles:
- **Photorealistic**: Professional photography quality
- **Digital Art**: Modern digital illustration
- **Oil Painting**: Classical art style
- **Anime**: Japanese animation style
- **Cyberpunk**: Futuristic neon aesthetic
- **Fantasy**: Epic fantasy art
- **Sketch**: Hand-drawn pencil style
- **3D Render**: CGI/Unreal Engine quality

## 📊 Performance

### GPU Requirements

| GPU | VRAM | Resolution | Speed |
|-----|------|------------|-------|
| RTX 3060 | 12GB | 512x512 | ~15s |
| RTX 3080 | 10GB | 768x768 | ~20s |
| RTX 4090 | 24GB | 1024x1024 | ~10s |

### CPU Mode
- Possible but very slow (5-10 minutes per image)
- Not recommended for regular use

## 🔧 Advanced Features

### Custom Model

```python
generator = StableDiffusionGenerator(
    model_id="stabilityai/stable-diffusion-xl-base-1.0"
)
```

### Image-to-Image

```python
from app import ImageEditor

editor = ImageEditor()

# Transform existing image
output = editor.img2img(
    image=input_image,
    prompt="turn this into a painting",
    strength=0.75
)
```

### Inpainting

```python
# Fill masked regions
output = editor.inpaint(
    image=original,
    mask=mask_image,
    prompt="a red apple"
)
```

## 💾 Generation History

All generations are automatically saved to `generations/` directory:
- `YYYYMMDD_HHMMSS_0.png`: Generated image
- `YYYYMMDD_HHMMSS_0.json`: Metadata (prompt, settings)

## 🐛 Troubleshooting

**Issue: Out of memory**
```
Solution: Reduce image size or use CPU
generator = StableDiffusionGenerator()
generator.device = "cpu"
```

**Issue: Slow generation**
```
- Use GPU if available
- Reduce inference steps to 30-40
- Use smaller dimensions (512x512)
```

**Issue: Poor quality images**
```
- Increase inference steps to 75-100
- Adjust guidance scale to 7-10
- Use quality tags in prompt
- Try negative prompts
```

**Issue: Model download fails**
```
# Manual download
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-diffusion-2-1")
```

## 🔐 Safety & Ethics

- Model includes safety checker (can be disabled for research)
- Please use responsibly
- Don't generate harmful, illegal, or copyrighted content
- Consider ethical implications of AI-generated media

## 📈 Tips for Best Results

1. **Be Specific**: More details = better results
   - ❌ "a cat"
   - ✅ "a fluffy orange tabby cat sitting on a windowsill"

2. **Use Quality Tags**: 
   - "highly detailed", "8k uhd", "professional"

3. **Specify Style**: 
   - "oil painting", "photograph", "digital art"

4. **Lighting**: 
   - "golden hour", "dramatic lighting", "soft ambient light"

5. **Negative Prompts**: 
   - Remove unwanted elements
   - "blurry, low quality, bad anatomy"

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Cloud Deployment

Works on:
- AWS EC2 (g4dn.xlarge or better)
- Google Cloud (n1-standard-4 + T4 GPU)
- RunPod, Vast.ai (GPU instances)

## 🤝 Contributing

Contributions welcome! Ideas:
- [ ] Add more preset styles
- [ ] Support for ControlNet
- [ ] Video generation
- [ ] Style mixing
- [ ] Batch processing
- [ ] API endpoint

## 📝 License

MIT License

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion
- Hugging Face for diffusers library
- Community prompt engineers

## 📧 Contact

For questions, open an issue on GitHub.

---

**Note**: First run requires downloading ~5GB of model weights. Subsequent runs are instant!
