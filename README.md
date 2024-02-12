# Advanced-Machine-Learning-CSE642-Assignments

## About
- Assignments and Course Work done in AML (CSE642) at Indraprastha Institute of Information Technology - Delhi (IIIT-D).
- Team: Aditya Ahuja (2020275) & Deeptanshu Barman Chowdhuri (2020293)
- Course Instructor: Dr. Gautam Shroff 
- Course Website: https://sites.google.com/view/advanced-ml/home.

## Milestone-1 Update

| **Dataset** | **Training target** |                 **Method Used**                 |
|:-----------:|:-------------------:|:-----------------------------------------------:|
|   0 Noise   |         Era         |                    Simple MLP                   |
|  Low Noise  |         Era         |                    Simple MLP                   |
|  Low Noise  |         Era         | Feature Subsets reconstructed using Autoencoder |
|  Low Noise  |    target_10_val    |       Noise Attention Loss Implimentation       |
|  High Noise |         Era         |                    Simple MLP                   |
|  High Noise |     taret_10_val    |             NAL with Feature Subsets            |

- Training, Validation and Calibration curves can be found in the ipynb notebooks
- Apart from the above methods, an ensemble model using MLPs trained on a subset of the training data was also created.
- An implimentation of this [method](https://arxiv.org/pdf/1803.09050.pdf) was also tested. The model however, is not currently functional.


