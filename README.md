# Everyday Object Classifier

![GitHub](https://img.shields.io/github/license/sparklinstar/Everyday_object_classification)
![GitHub stars](https://img.shields.io/github/stars/sparklinstar/Everyday_object_classification?style=social)
![GitHub forks](https://img.shields.io/github/forks/sparklinstar/Everyday_object_classification?style=social)
![GitHub issues](https://img.shields.io/github/issues/sparklinstar/Everyday_object_classification)

An EfficientNet-based image classification model deployed on Streamlit, capable of identifying everyday objects from various categories. The model is trained on a custom dataset compiled from three sources: [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals), [Vehicle Images Dataset](https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset), and [Daily Objects around the World Dataset](https://www.kaggle.com/datasets/humansintheloop/dollar-street-dataset).

## Deployed Application

Access the deployed application and interact with the model: [Everyday Object Classifier](https://everyday-object-classifier.streamlit.app/)

## Supported Labels

The model classifies the following labels:
- Big Truck
- City Car
- Multi Purpose Vehicle
- Sedan
- Sport Utility Vehicle
- Truck
- Van
- Cat
- Caterpillar
- Cockroach
- Cow
- Dog
- Rat
- Refrigerators
- Sofas
- TVs
- Wall Clocks

## Key Features

- Data preprocessing and augmentation using TensorFlow's ImageDataGenerator
- Transfer learning with EfficientNetB2 architecture
- Fine-tuning layers for improved performance
- Early stopping and learning rate reduction during training
- Model evaluation with accuracy, F1 score, classification report, and confusion matrix
- Streamlit deployment for user-friendly interaction

## How to Use

1. Clone the repository: `git clone https://github.com/yourusername/everyday-object-classifier.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure dataset paths in the script to match your environment.
4. Run the script to train the model and visualize its performance.
5. Explore the deployed application to classify everyday objects.

## Dataset Sources

- [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
- [Vehicle Images Dataset](https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset)
- [Daily Objects around the World Dataset](https://www.kaggle.com/datasets/humansintheloop/dollar-street-dataset)


## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to explore, modify, and adapt this project for your own use. If you find it useful, don't forget to star the repository and provide feedback!