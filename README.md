# DL-NumRecognizer
A deep learning-based project for real-time digit recognition and visualization using PyTorch and OpenCV. This application trains a neural network to recognize handwritten digits with simple live visualization of predictions, allowing users to observe model performance as it improves over training epochs.

## Features
- **Real-Time Training Visualization**: Watch the model's predictions live as it learns to classify handwritten digits. Visualization can be enabled or disabled with a command-line flag.
- **Interactive Digit Recognition**: Test the trained model with your own input for real-time feedback on digit classification accuracy.
- **Customizable Training**: Easily adjust training parameters, including batch size, learning rate, and epoch count, to experiment with different setups.

## Screenshots
<img src="https://github.com/user-attachments/assets/f4112d75-6f89-4f92-8043-2f7075897ea4" width="600px"><br>
*Real-time visualization of model predictions during training.*

<img src="https://github.com/user-attachments/assets/e693135a-cf6b-4f3b-9836-01bd8dfaf43d" width="600px"><br>
*Test the model's digit recognition capabilities interactively.*

## How to Run
1. **Install Dependencies**: Install the required libraries:
   ```bash
   pip install torch torchvision opencv-python matplotlib
   ```
   > **Note**: If you have an NVIDIA GPU, installing CUDA is recommended for faster training. To enable CUDA, install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) (optional, but recommended). PyTorch will automatically detect CUDA if it is properly configured. You can find more instructions in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
   > Find the suitable PyTorch version for your CUDA version, and install it using a command like the following:
   > ```bash
   > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   > ```

2. **Train the Model**: You can either train your own model or download a pre-trained model. Visualization is optional and can be enabled by adding the `--visualize` flag:
   ```bash
   python train.py --visualize
   ```

3. **Evaluate the Model**: After training (or downloading) the model, you can interact with it to classify handwritten digits. Run the following command:
   ```bash
   python predict.py
   ```
   - In the drawing window:
     - **Press `p` or `Enter`** to classify the drawn digit. The predicted digit, along with the model's confidence, will be displayed in the console window.
     - **Press `q` or `Esc`** to close the drawing window and exit the program.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.
