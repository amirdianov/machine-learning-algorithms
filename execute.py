from config.conf import cfg

from datasets.digits_dataset import Digits
from utils.enums import SetType

train = Digits(cfg)(SetType.train)
valid = Digits(cfg)(SetType.valid)
test = Digits(cfg)(SetType.test)

print(train, valid, test)
