from utils import MODEL_NAME
from simplifier import Simplifier

simplifier = Simplifier(MODEL_NAME)
text = '14 декабря 1944 года рабочий посёлок Ички был переименован в рабочий посёлок Советский, после чего поселковый совет стал называться Советским.'
print(simplifier.simplify(text))