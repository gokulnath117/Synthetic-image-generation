1. Install Required Libraries
   for python
   pip install torch torchvision diffusers transformers opencv-python pillow matplotlib

   For julia
   using Pkg
    Pkg.add(["Flux", "Images", "ImageIO", "ImageTransformations", "CUDA"])

Approach for Each Task
  Synthetic Image Generation
    Used the diffusers library to load a pre-trained Stable Diffusion model. 
    Generated three synthetic images from a single text prompt.
    Saved the generated images to disk with appropriate filenames.

  Image Preprocessing
    Resized images to a fixed size of 224×224 pixels.
    Converted them to grayscale (optional simplification step).
    Transformed them into tensors, ensuring values are normalized between 0 and 1.

  Minimal Neural Network (Forward Pass)
    Implemented a simple CNN model using PyTorch.
    Loaded a preprocessed image and converted it to the correct tensor shape.
    Passed the image through the CNN to obtain an output prediction.
    Printed the output to verify successful execution.

Challenges

Lack of built-in CUDA: My local machine does not have CUDA support, so I used Kaggle to execute GPU-intensive tasks

   
