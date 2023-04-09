from config.conf import cfg
from datasets.digits_dataset import Digits
from desition_tree_exmpl import DT
from utils.enums import SetType, SetTypeOfTask

train = Digits(cfg)(SetType.train)
valid = Digits(cfg)(SetType.valid)
test = Digits(cfg)(SetType.test)

des_tr = DT(SetTypeOfTask.classification.name)
des_tr.train(train['inputs'], train['targets'])
des_tr.get_predictions(valid['inputs'])
