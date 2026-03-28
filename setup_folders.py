import os
STRUCTURE = {
    'data' : ['raw_images' , 'processed_images' , 'filtered_images' , 'uploads'],
    'src' : [],
    'api' : [],
    'notebooks' : [],
    'tests' : [],
    'results': ['plots', 'embeddings', 'models', 'metrics'],
}
for parent, children in STRUCTURE.items():
    os.makedirs(parent, exist_ok=True)
    for child in children:
        os.makedirs('f{parent}/{child}', exist_ok=True)
    if parent in ['src' , 'api' , 'tests']:
        init_path = os.path.join(parent, '__init__.py')
        open(f'{parent}/__init__.py' , 'a').close()
print("Folder structure created:")
for parent in STRUCTURE:
    print(f' {parent}/')