"""
Verify that all critical libraries are working
Run this before Day 1 starts
"""

def verify_pytorch():
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    
def verify_opencv():
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    
def verify_segmentation_libs():
    try:
        import segmentation_models_pytorch as smp
        print(f"segmentation-models-pytorch: OK")
    except:
        print("segmentation-models-pytorch: NOT INSTALLED")
    
    try:
        import detectron2
        print(f"detectron2: OK")
    except:
        print("detectron2: NOT INSTALLED (optional)")

def test_image_loading():
    import cv2
    import os
    
    test_dir = "data/raw/test_samples"
    if not os.path.exists(test_dir):
        print(f"Create {test_dir} and add sample images")
        return
    
    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    if len(images) == 0:
        print("No test images found")
        return
    
    img = cv2.imread(os.path.join(test_dir, images[0]))
    print(f"Successfully loaded test image: {images[0]}, shape: {img.shape}")

def test_pretrained_model():
    import torch
    import torchvision.models as models
    
    model = models.resnet50(pretrained=True)
    print("Successfully loaded pretrained ResNet50")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

if __name__ == "__main__":
    print("="*50)
    print("ENVIRONMENT VERIFICATION")
    print("="*50)
    
    verify_pytorch()
    print("-"*50)
    verify_opencv()
    print("-"*50)
    verify_segmentation_libs()
    print("-"*50)
    test_image_loading()
    print("-"*50)
    test_pretrained_model()
    print("-"*50)
    print("✓ Setup verification complete!")