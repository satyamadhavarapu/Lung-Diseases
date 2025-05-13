import unittest
import numpy as np
import os
from app import  predict_image, count_images  # Replace with your filename

class TestLungDiseaseModel(unittest.TestCase):
    
    def test_model_build(self):
        num_classes = 4  # Example
        model = build_custom_model(num_classes)
        self.assertEqual(model.output_shape[-1], num_classes)
        
    def test_image_prediction(self):
        # Assuming you have a small dummy image
        temp_img = np.random.rand(224, 224, 3)
        temp_path = "temp_test_img.jpg"
        import matplotlib.pyplot as plt
        plt.imsave(temp_path, temp_img)
        
        predicted_class, probabilities = predict_image(temp_path)
        self.assertEqual(len(probabilities), 4)  # Should match number of classes
        
        os.remove(temp_path)
        
    def test_count_images(self):
        os.makedirs('test_dir/class1', exist_ok=True)
        open('test_dir/class1/image1.jpg', 'a').close()
        open('test_dir/class1/image2.jpg', 'a').close()
        
        counts = count_images('test_dir')
        self.assertEqual(counts['class1'], 2)
        
        # Cleanup
        os.remove('test_dir/class1/image1.jpg')
        os.remove('test_dir/class1/image2.jpg')
        os.rmdir('test_dir/class1')
        os.rmdir('test_dir')

if __name__ == '__main__':
    unittest.main()
