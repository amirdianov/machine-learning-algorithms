from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier

dataset = Sportsmanheight()()
predictions = Classifier()(dataset['height'])
gt = dataset['class']
