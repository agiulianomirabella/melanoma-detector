from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData
from root.topology.persistence import display_persistent_diagram, extractPoints
import tadasets


img = readData(1)[0][0]
data = extractPoints(img)
display_persistent_diagram(data)


'''
loop = tadasets.dsphere(d= 1)
print(loop.shape)
display_persistent_diagram(loop)
'''