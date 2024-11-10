import numpy as np
import cv2
import torch
import torch.nn.functional as F
from model import SimpleNet

class DigitRecognizer:
    def __init__(self, model_path='model.pth'):
        self.canvas = np.ones((280, 280), dtype="uint8") * 255
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self):
        model = SimpleNet().to(self.device)
        model.load_state_dict(torch.load(self.model_path, weights_only=True))
        model.eval()
        return model

    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(self.canvas, (x, y), 10, (0,), -1)

    def preprocess_image(self):
        img = cv2.resize(self.canvas, (28, 28))
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        cx, cy = x + w // 2, y + h // 2
        shift_x, shift_y = 14 - cx, 14 - cy
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (28, 28))

        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)

    def predict_digit(self):
        img = self.preprocess_image()
        with torch.no_grad():
            output = self.model(img)
            probabilities = F.softmax(output, dim=1)
            certainty, predicted_digit = torch.max(probabilities, 1)
            return predicted_digit.item(), certainty.item() * 100

    def reset_canvas(self):
        self.canvas[:] = 255

    def run(self):
        cv2.namedWindow("Digit Recognizer")
        cv2.setMouseCallback("Digit Recognizer", self.draw)

        while True:
            cv2.imshow("Digit Recognizer", self.canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q")  or key == 27:
                break
            elif key == ord("p") or key == 13:
                digit, certainty = self.predict_digit()
                print(f"Predicted digit: {digit} (certainty: {certainty:.2f}%)")
                self.reset_canvas()
                cv2.imshow("Digit Recognizer", self.canvas)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

def main():
    recognizer = DigitRecognizer()
    recognizer.run()

if __name__ == '__main__':
    main()