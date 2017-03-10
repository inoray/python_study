import pandas as pd
import pandas.io.data as web
import datetime
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import *


start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2015, 2, 15)
soil = web.DataReader("010950.KS", "yahoo", start, end)

print(soil)

#plt.plot(soil.index, soil['Close'])
#plt.show()

app = QApplication(sys.argv)
label = QLabel("Hello, PyQt")
label.show()

print("Before event loop")
app.exec_()
print("After event loop")