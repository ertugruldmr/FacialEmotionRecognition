import gradio as gr
import tensorflow as tf
import numpy as np

# loading the files
model_path = "Basic_EmotionModel"
model = tf.keras.models.load_model(model_path)
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Util Functions
def process_image(image):
    # Convert into tensor
    image = tf.convert_to_tensor(image)

    # Cast the image to tf.float32
    image = tf.cast(image, tf.float32)
    
    # Resize the image to img_resize
    image = tf.image.resize(image, (64,64))
    
    # Normalize the image
    image /= 255.0
    
    # Return the processed image and label
    return image

def predict(image):

  # Pre-procesing the data
  images = process_image(image)

  # Batching
  batched_images = tf.expand_dims(images, axis=0)
  
  prediction = model.predict(batched_images).flatten()
  confidences = {labels[i]: np.round(float(prediction[i]), 7) for i in range(len(labels))}
  return confidences



# declerating the params
demo = gr.Interface(fn=predict, 
             inputs=gr.Image(shape=(32, 32)),
             outputs=gr.Label(num_top_classes=len(labels)),
             examples="sample_images")
            
            
# Launching the demo
if __name__ == "__main__":
    demo.launch()
