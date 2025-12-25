"""
GenAI Text-to-Image System
Generate images from text prompts using Stable Diffusion and other models
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional
import io
import json
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StableDiffusionGenerator:
    """Stable Diffusion image generation"""
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1"):
        """Initialize Stable Diffusion model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load text-to-image pipeline
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        # Optimize scheduler
        self.txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.txt2img_pipe.scheduler.config
        )
        
        self.txt2img_pipe = self.txt2img_pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.txt2img_pipe.enable_attention_slicing()
            self.txt2img_pipe.enable_vae_slicing()
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            num_images: Number of images to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: How strictly to follow prompt (1-20)
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL Images
        """
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        logger.info(f"Generating {num_images} image(s) with prompt: {prompt[:50]}...")
        
        # Generate images
        with torch.autocast(self.device):
            output = self.txt2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        images = output.images
        logger.info(f"Generated {len(images)} image(s)")
        
        return images
    
    def enhance_prompt(self, simple_prompt: str) -> str:
        """
        Enhance a simple prompt with quality tags
        
        Args:
            simple_prompt: Basic prompt
            
        Returns:
            Enhanced prompt
        """
        quality_tags = [
            "highly detailed",
            "professional photography",
            "8k uhd",
            "high quality",
            "sharp focus",
            "trending on artstation"
        ]
        
        enhanced = f"{simple_prompt}, {', '.join(quality_tags)}"
        return enhanced


class ImageEditor:
    """Image-to-image and inpainting capabilities"""
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1"):
        """Initialize image editing pipelines"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load img2img pipeline
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.img2img_pipe = self.img2img_pipe.to(self.device)
        
        # Load inpainting pipeline
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.inpaint_pipe = self.inpaint_pipe.to(self.device)
        
        logger.info("Image editing models loaded")
    
    def img2img(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> Image.Image:
        """
        Transform existing image based on prompt
        
        Args:
            image: Input PIL Image
            prompt: Transformation prompt
            strength: How much to transform (0-1)
            guidance_scale: Guidance strength
            num_inference_steps: Number of steps
            
        Returns:
            Transformed image
        """
        logger.info(f"Transforming image with prompt: {prompt[:50]}...")
        
        with torch.autocast(self.device):
            output = self.img2img_pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        
        return output.images[0]
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> Image.Image:
        """
        Inpaint masked regions
        
        Args:
            image: Original image
            mask: Binary mask (white = inpaint)
            prompt: What to generate in masked area
            guidance_scale: Guidance strength
            num_inference_steps: Number of steps
            
        Returns:
            Inpainted image
        """
        logger.info(f"Inpainting with prompt: {prompt[:50]}...")
        
        with torch.autocast(self.device):
            output = self.inpaint_pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        
        return output.images[0]


class PromptLibrary:
    """Library of example prompts"""
    
    STYLES = {
        "Photorealistic": "photorealistic, 8k uhd, high resolution, professional photography",
        "Digital Art": "digital art, concept art, trending on artstation, highly detailed",
        "Oil Painting": "oil painting, classical art, fine art, museum quality",
        "Anime": "anime style, manga, japanese animation, vibrant colors",
        "Cyberpunk": "cyberpunk, neon lights, futuristic, sci-fi, blade runner",
        "Fantasy": "fantasy art, magical, mystical, epic, dramatic lighting",
        "Sketch": "pencil sketch, hand drawn, artistic, detailed line art",
        "3D Render": "3d render, octane render, unreal engine, cgi, hyperrealistic"
    }
    
    SUBJECTS = {
        "Portrait": "portrait of a person, face focus, detailed features",
        "Landscape": "beautiful landscape, scenic view, nature",
        "Architecture": "architectural design, building, structure",
        "Still Life": "still life, objects arrangement, composition",
        "Abstract": "abstract art, geometric shapes, colors",
        "Character": "character design, full body, dynamic pose",
        "Scene": "detailed scene, environment, atmosphere"
    }
    
    NEGATIVE_PROMPTS = [
        "blurry, low quality, bad anatomy, bad proportions",
        "ugly, distorted, deformed, disfigured",
        "watermark, text, signature, logo",
        "oversaturated, noise, grain, artifacts"
    ]
    
    @classmethod
    def get_style_prompt(cls, subject: str, style: str) -> Tuple[str, str]:
        """Get styled prompt with negative prompt"""
        style_modifier = cls.STYLES.get(style, "")
        negative = ", ".join(cls.NEGATIVE_PROMPTS)
        
        prompt = f"{subject}, {style_modifier}"
        
        return prompt, negative


class GenerationHistory:
    """Track generation history"""
    
    def __init__(self, save_dir: str = "generations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.history = []
    
    def save_generation(
        self,
        images: List[Image.Image],
        prompt: str,
        params: dict
    ) -> List[str]:
        """Save generated images and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_paths = []
        for i, image in enumerate(images):
            # Save image
            filename = f"{timestamp}_{i}.png"
            filepath = self.save_dir / filename
            image.save(filepath)
            saved_paths.append(str(filepath))
            
            # Save metadata
            metadata = {
                "prompt": prompt,
                "timestamp": timestamp,
                "params": params,
                "filename": filename
            }
            
            metadata_file = self.save_dir / f"{timestamp}_{i}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.history.append({
            "timestamp": timestamp,
            "prompt": prompt,
            "files": saved_paths
        })
        
        return saved_paths


def main():
    """Streamlit UI"""
    st.set_page_config(
        page_title="GenAI Text-to-Image",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 GenAI Text-to-Image Generator")
    st.markdown("Generate stunning images from text descriptions using Stable Diffusion")
    
    # Initialize session state
    if 'generator' not in st.session_state:
        with st.spinner("Loading Stable Diffusion model... This may take a few minutes."):
            st.session_state.generator = StableDiffusionGenerator()
            st.session_state.history = GenerationHistory()
        st.success("Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        # Style selection
        style = st.selectbox("Style", list(PromptLibrary.STYLES.keys()))
        
        # Basic parameters
        width = st.select_slider("Width", options=[256, 384, 512, 640, 768], value=512)
        height = st.select_slider("Height", options=[256, 384, 512, 640, 768], value=512)
        
        num_images = st.slider("Number of Images", 1, 4, 1)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            num_steps = st.slider("Inference Steps", 20, 100, 50)
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
            seed = st.number_input("Seed (-1 for random)", -1, 999999, -1)
            use_negative = st.checkbox("Use Negative Prompt", value=True)
            enhance_prompt = st.checkbox("Auto-enhance Prompt", value=True)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Prompts")
        
        if st.button("🎭 Fantasy Character"):
            st.session_state.quick_prompt = "a majestic fantasy character with magical powers"
        if st.button("🏞️ Landscape"):
            st.session_state.quick_prompt = "a beautiful mountain landscape at sunset"
        if st.button("🚀 Sci-Fi"):
            st.session_state.quick_prompt = "futuristic spaceship in deep space"
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Prompt")
        
        # Get quick prompt if available
        default_prompt = st.session_state.get('quick_prompt', '')
        
        prompt = st.text_area(
            "Describe what you want to generate:",
            value=default_prompt,
            height=100,
            placeholder="A serene Japanese garden with cherry blossoms..."
        )
        
        if use_negative:
            negative_prompt = st.text_area(
                "Negative prompt (what to avoid):",
                value=", ".join(PromptLibrary.NEGATIVE_PROMPTS),
                height=80
            )
        else:
            negative_prompt = ""
        
        # Generate button
        if st.button("🎨 Generate Images", type="primary", use_container_width=True):
            if prompt:
                with st.spinner("Generating images... This may take a minute."):
                    try:
                        # Enhance prompt if requested
                        if enhance_prompt:
                            final_prompt, auto_negative = PromptLibrary.get_style_prompt(prompt, style)
                            if not use_negative:
                                negative_prompt = auto_negative
                        else:
                            final_prompt = prompt
                        
                        # Generate
                        images = st.session_state.generator.generate(
                            prompt=final_prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_images=num_images,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale,
                            seed=None if seed == -1 else seed
                        )
                        
                        # Save to history
                        params = {
                            "style": style,
                            "width": width,
                            "height": height,
                            "steps": num_steps,
                            "guidance": guidance_scale,
                            "seed": seed
                        }
                        saved_paths = st.session_state.history.save_generation(
                            images, final_prompt, params
                        )
                        
                        # Store in session
                        st.session_state.generated_images = images
                        st.session_state.current_prompt = final_prompt
                        
                        st.success(f"✅ Generated {len(images)} image(s)!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating images: {str(e)}")
            else:
                st.warning("⚠️ Please enter a prompt first!")
    
    with col2:
        st.subheader("🖼️ Generated Images")
        
        if 'generated_images' in st.session_state:
            images = st.session_state.generated_images
            
            # Display images
            for i, img in enumerate(images):
                st.image(img, caption=f"Image {i+1}", use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                st.download_button(
                    label=f"⬇️ Download Image {i+1}",
                    data=buf.getvalue(),
                    file_name=f"generated_{i+1}.png",
                    mime="image/png"
                )
            
            st.markdown(f"**Prompt used:** {st.session_state.current_prompt}")
        else:
            st.info("👈 Configure settings and click 'Generate Images' to create AI art!")
    
    # History section
    with st.expander("📜 Generation History"):
        if st.session_state.history.history:
            for item in reversed(st.session_state.history.history[-10:]):
                st.markdown(f"**{item['timestamp']}** - {item['prompt'][:100]}...")
        else:
            st.info("No generation history yet")


if __name__ == "__main__":
    main()
