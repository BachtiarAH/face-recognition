import kivy
kivy.require('2.1.0')  # Replace with the latest version if needed

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import tensorflow as tf

class CameraApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.model = tf.lite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()[0]
        self.output_details = self.model.get_output_details()[0]

        # Define label mapping
        self.labels = ["bachtiar", "masker", "nothing"]

        # Create layout and widgets
        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.prediction_label = Label(text="Prediction: None", size_hint_y=None, height=40)
        
        layout.add_widget(self.image)
        layout.add_widget(self.prediction_label)

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 fps
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Preprocess the frame
            input_data = cv2.resize(frame, (self.input_details['shape'][2], self.input_details['shape'][1]))
            input_data = input_data.astype('float32') / 255.0
            input_data = input_data.reshape(self.input_details['shape'])
            
            # Run inference
            self.model.set_tensor(self.input_details['index'], input_data)
            self.model.invoke()
            output_data = self.model.get_tensor(self.output_details['index'])
            
            # Get the index of the highest probability
            predicted_index = output_data[0].argmax()
            predicted_label = self.labels[predicted_index]
            
            # Update the prediction label
            self.prediction_label.text = f"Prediction: {predicted_label} ({predicted_index})"

            # Convert the frame to a format Kivy can use
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr')
            self.image.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    CameraApp().run()
