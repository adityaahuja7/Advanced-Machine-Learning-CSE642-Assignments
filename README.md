# Advanced Machine Learning (AML Winter'24) Assignment-1

**Authors:** Aditya Ahuja, Deeptanshu Barman

**Date:** March 16, 2024

## Milestone-1 Update

### Folder Structure

- All models run on the "era" dataset can be found in the notebooks under the "feature noise" folder.
- All models run on the "target 10 val" dataset can be found in the notebooks under the "Feature and Label Noise" folder.

### Running the Models

- Each notebook can be run multiple times for different noise scenarios by selecting the data frame at the beginning of the notebook. (See the configuration cell example in the notebooks.)

### Notes & Observations

- Most models were trained for 20 epochs with a batch size of 128.
- We trained autoencoder models to reconstruct the entire feature set from feature subsets (paper). The autoencoder was trained using the reconstruction loss for the entire feature set. We detached the encoder from the trained autoencoder, attached a simple classification head, and fine-tuned the model to generate probability scores for the respective classes.
- We used the Noise Attention Loss (NAL) in place of simple cross-entropy loss to improve performance in the noisy label setting, including with our Feature Subset model.
- Some models (specifically our custom ensemble function and Feature Subset classification model) did not perform well and will be improved before the next milestone.

### Results

| Dataset    | Training target | Method Used                   | Training Accuracy | Validation Accuracy |
|------------|-----------------|--------------------------------|-------------------|---------------------|
| 0 Noise    | Era             | Simple MLP                    | 95.06             | 95.59               |
| Low Noise  | Era             | Simple MLP                    | 60.66             | 66.58               |
| Low Noise  | Era             | CEFS                          | 54.85             | 54.69               |
| Low Noise  | target 10 val   | Simple MLP                    | 78.29             | 78.43               |
| Low Noise  | target 10 val   | NAL                           | 65.10             | 65.47               |
| Low Noise  | target 10 val   | CEFS with NAL                 | 74.12             | 73.79               |
| High Noise | Era             | Simple MLP                    | 40.55             | 45.09               |
| High Noise | Era             | CEFS                          | 39.69             | 39.51               |
| High Noise | target 10 val   | Simple MLP                    | 61.41             | 61.35               |
| High Noise | target 10 val   | NAL                           | 64.87             | 65.46               |
| High Noise | target 10 val   | CEFS with NAL                 | 58.72             | 58.76               |

Table 1: Milestone-1 Results. *CEFS: Classification using an encoded representation of Feature Subsets*
*NAL: Noise Attention Loss.*

## Milestone-2 Update

### Models Trained & Observations

- **TabPFN:** We trained TabPFN models for both the "target 10 val" and "era" attributes. The main constraints we faced with TabPFN were a row limit of <= 1000 and a class limit of <= 10. For "target 10 val," the constraint was only on the number of rows, which we addressed by randomly subsampling n-models from the data. This approach yielded better accuracy than any single TabPFN.
- **Sequential Models:** We trained Long Short-Term Memory (LSTM) models on Sequential Low Noise & High Noise data. We sorted the data by the row number and trained the LSTM model to classify "target 10 val" of sequences of size 20. We used a batch size of 1024 and a learning rate of 0.01 for the first 20 epochs and 0.003 for the last 5 epochs. We also experimented with Transformers to generate an encoded representation of the input sequence and further trained a linear classifier to classify using this representation.

### Results

| Dataset    | Training Target | Method Used                 | Training Accuracy | Validation Accuracy |
|------------|-----------------|------------------------------|-------------------|---------------------|
| 0 Noise    | Era             | TabPFN                      | N/A               | 85.83%              |
| Low Noise  | Era             | TabPFN                      | N/A               | 55%                 |
| High Noise | Era             | TabPFN                      | N/A               | 45%                 |
| Low Noise  | Target 10 val   | TabPFN                      | N/A               | 77.92%              |
| Low Noise  | Target 10 val   | Classification using Transformer | 77.63%       | 78.40%              |
| Low Noise  | Target 10 val   | Classification using LSTM   | 71.70%            | 71.62%              |
| High Noise | Target 10 val   | TabPFN                      | N/A               | 61.19%              |
| High Noise | Target 10 val   | Classification using Transformer | 64.02%       | 65.29%              |
| High Noise | Target 10 val   | Classification using LSTM   | 55.63%            | 56.05%              |

Table 2: Milestone-2 Results

## Milestone-3 Update

### Notes & Observation

- Trained cascaded MLP models to make conservative predictions on Noisy Features as well as Noisy Features and Labels.
- Performed calibration by minimizing the expected calibration error (ECE) along with the Cross Entropy Loss. The loss function was a sum of both the CEL and the ECE with weights of both being equal to 1. We used temperature scaling by incorporating a temperature parameter in the softmax layer.
- Performed selective classification using cascading along with NAL Loss. We fixed the number of cascade levels to 4 and ensured that the model was predicting for more than 60% for both cases.
- As the cascade levels increase, we increase the model complexity by increasing the number of neurons in the hidden layers. For example, in the first level, the hidden layers are [24, 64, 64, 12]; in the second level, the hidden layers are [24, 128, 128, 12]; and so on.
- Pruned the datapoints based upon the gini impurity at each level. The impurity levels were set as follows: [0.1, 0.2, 0.3, 0.4].

### Results

| Noise Level | Target        | Method                                        | Training Accuracy | Test Accuracy |
|-------------|----------------|------------------------------------------------|-------------------|---------------|
| Low         | Era           | MLP with Calibration                          | 0.7623            | 0.7535        |
| Low         | Era           | Cascading MLP with Calibration                | N/A               | 0.9432        |
| Low         | Era           | Cascading MLP with NAL Loss and Calibration   | N/A               | 0.9281        |
| High        | Era           | MLP with Calibration                          | 0.5136            | 0.5109        |
| High        | Era           | Cascading MLP with Calibration                | N/A               | 0.9275        |
| High        | Era           | Cascading MLP with NAL Loss and Calibration   | N/A               | 0.8953        |
| Low         | Target 10 Val | MLP with Calibration                          | 0.8115            | 0.8171        |
| Low         | Target 10 Val | Cascading MLP with Calibration                | N/A               | 0.9507        |
| Low         | Target 10 Val | Cascading MLP with NAL Loss and Calibration   | N/A               | 0.9660        |
| High        | Target 10 Val | MLP with Calibration                          | 0.6381            | 0.6393        |
| High        | Target 10 Val | Cascading MLP with Calibration                | N/A               | 0.9222        |
| High        | Target 10 Val | Cascading MLP with NAL Loss and Calibration   | N/A               | 0.9459        |

Table 3: Milestone-3 Results

## Revisiting Milestone-1 Results

We explored the reason behind our relatively low accuracies in Milestone-1 and discovered that the penultimate layer was passing through a ReLU activation prior to softmax, which was resulting in the model not learning the correct underlying pattern. After removing this bug from our code, the updated results are as follows:

| Dataset    | Training target | Method Used                   | Training Accuracy | Validation Accuracy |
|------------|-----------------|--------------------------------|-------------------|---------------------|
| 0 Noise    | Era             | Simple MLP                    | 95.06             | 93.59               |
| Low Noise  | Era             | Simple MLP                    | 77.05             | 76.66               |
| Low Noise  | Era             | CEFS                          | 54.85             | 54.69               |
| Low Noise  | target 10 val   | Simple MLP                    | 82.39             | 81.99               |
| Low Noise  | target 10 val   | NAL                           | 65.10             | 65.47               |
| Low Noise  | target 10 val   | CEFS with NAL                 | 71.63             | 71.44               |
| High Noise | Era             | Simple MLP                    | 51.69             | 52.30               |
| High Noise | Era             | CEFS                          | 39.69             | 39.51               |
| High Noise | target 10 val   | Simple MLP                    | 64.92             | 64.53               |
| High Noise | target 10 val   | NAL                           | 64.87             | 65.46               |
| High Noise | target 10 val   | CEFS with NAL                 | 52.00             | 51.98               |

Table 4: Improved results for Milestone-1
