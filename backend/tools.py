from langchain.tools import StructuredTool
import requests
from pydantic import BaseModel, Field
from typing import Optional
import logging

# Note: This isn't functional, but should be the basis for multiple inputs for image generation.
class GenerateImageInput(BaseModel):
    prompt: str = Field(..., description="The prompt to generate an image from.")
    negative: Optional[str] = Field("", description="Negative prompt to avoid certain features in the image.")
    images: Optional[int] = Field(1, description="Number of images to generate.")
    steps: Optional[int] = Field(50, description="Number of steps for image generation.")
    cfgScale: Optional[float] = Field(3.0, description="Classifier-free guidance scale.")
    seed: Optional[int] = Field(-1, description="Seed for random number generation, -1 for random.")
    model: Optional[str] = Field("hyphoriaIllustrious20_v001", description="Model name to use.")
    width: Optional[int] = Field(1024, description="Width of the generated base image before upscale.")
    height: Optional[int] = Field(1024, description="Height of the generated base image before upscale.")
    refinerControlPercentage: Optional[float] = Field(0.0, description="Percentage of image refinement, 0.0 for no refinement, 1.0 for full change.")
    refinerUpscale: Optional[float] = Field(2.0, description="Refiner upscale factor. Final resolution will be width x refinerUpscale and height x refinerUpscale.")
    refinerMethod: Optional[str] = Field("stepSwap", description="Refiner method to use, e.g., 'stepSwap', 'normal'")
    sampler: Optional[str] = Field("dpmpp_3m_sde_gpu", description="Sampler to use for image generation, e.g., 'dpmpp_3m_sde_gpu', 'dpmpp_2m_sde_gpu', etc")
    scheduler: Optional[str] = Field("simple", description="Scheduler to use for image generation, e.g., 'karras', 'euler', 'simple', 'normal', 'exponential', etc")

# This function accepts GenerateImageInput args and calls the SwarmUI MCP server to generate an image
def generate_image(**kwargs) -> dict:
    """Generate an image from a text prompt and other optional values using the SwarmUI MCP server."""
    try:
        # Convert to dict for request
        input_model = GenerateImageInput(**kwargs)
        payload = input_model.dict()
        
        # Call the generate-image endpoint
        response = requests.post(
            "http://localhost:5001/generate-image",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error processing input")
        logging.error(f"Error details: {str(e)}")
        raise

# Create the StructuredTool for image generation from the generate_image function with GenerateImageInput args
generate_image = StructuredTool(
    func=generate_image,
    name="generate_image",
    description="A tool to generate an image from a text prompt using SwarmUI MCP server. Accepts optional negative prompt, steps, cfgScale, model, images, and seed.",
    args_schema=GenerateImageInput,
    return_direct=True
)