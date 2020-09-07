import os
from PIL import Image

annotation_path = "C:\\Users\\Henry Ng\\Desktop\\FLIR_ADAS\\validation\\output"
image_folder = "C:\\Users\\Henry Ng\\Desktop\\FLIR_ADAS\\validation\\PreviewData"
output_folder = "C:\\Users\\Henry Ng\\Desktop\\FLIR_ADAS\\validation\\testing_images\\infrared"

def extract_images(annotation_path, image_folder, output_folder):
    file_names = []
    for file in os.listdir(annotation_path):
        name_of_file = file.split(".")
        file_names.append(name_of_file[0])
    
    for image in os.listdir(image_folder):
        image_name = image.split(".")
        if(image_name[0] in file_names):
            im = Image.open(f'{image_folder}\\{image}')
            width, height = im.size
            
            with open(f'{annotation_path}\\{image_name[0]}.txt') as f:
                count = 0
                for line in f:
                    coord = line.split(" ")
                    
                    image_path = f"{output_folder}\\c{coord[0]}_{count}_{image_name[0]}.jpg"
                    
                    xcenter = float(coord[1]) * width 
                    ycenter = float(coord[2]) * height
                    box_width = float(coord[3]) * width
                    box_height = float(coord[4]) * height
                    
                    x1 = xcenter - box_width / 2 
                    y1 = ycenter - box_height / 2
                    x2 = xcenter + box_width / 2
                    y2 = ycenter + box_height / 2
                    
                    im1 = im.crop((x1, y1, x2, y2))
                    im1.save(image_path, 'JPEG')
                    count += 1
            
                    
extract_images(annotation_path, image_folder, output_folder)
                