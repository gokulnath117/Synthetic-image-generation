using Flux
using Images, ImageIO, ImageTransformations
using Flux: glorot_uniform


model = Chain(
    Conv((3, 3), 1 => 8, relu),  
    MaxPool((2, 2)),             
    Conv((3, 3), 8 => 16, relu),  
    MaxPool((2, 2)),             
    Flux.flatten,                
    Dense(576, 10, relu)         
)

println("Model Architecture:")
println(model)


img_path = "./processed_images/processed_image_1.png"
img = load(img_path)  

img_resized = imresize(img, (32, 32))  
img_gray = Gray.(img_resized)          
img_tensor = Float32.(channelview(img_gray))  


input_tensor = reshape(img_tensor, 32, 32, 1, 1) 

output = model(input_tensor)

println("Forward Pass Output:")
println(output)
