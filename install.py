import launch

packages = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
    "SAM": "segment-anything",
    "huggingface_hub": "huggingface_hub",
    "groundino": "groundingdino-py==0.4.0",
}

for name, target in packages.items():
    if not launch.is_installed(name):
        launch.run_pip(f'install {target}', desc=f'{name} for shop_tools')
