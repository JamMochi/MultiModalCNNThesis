import os 

main_path = '/home/w004hsn/Desktop/CNNTrainer'
inf = '1000_inf'
rgb = '1000_rgb'

rgb_files = os.listdir(os.path.join(main_path, rgb))
inf_files = os.listdir(os.path.join(main_path, inf))

for file in os.listdir(os.path.join(main_path, inf)):
    if file not in rgb_files:
        os.remove(file)
        print(f'{file} is removed')
    
for file in os.listdir(os.path.join(main_path, rgb)):
    if file not in inf_files:
        os.remove(file)
        print(f'{file} is removed')