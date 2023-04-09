from config.conf import cfg
from datasets.digits_dataset import Digits
from desition_tree_exmpl import DT
from utils.enums import SetType, SetTypeOfTask

train = Digits(cfg)(SetType.train)
valid = Digits(cfg)(SetType.valid)
test = Digits(cfg)(SetType.test)

des_tr = DT(train['inputs'].shape[0], SetTypeOfTask.classification.name)
des_tr.train(train['inputs'], train['targets'])
