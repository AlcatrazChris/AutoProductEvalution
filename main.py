from PyQt5.QtWidgets import QApplication
from UI import App
import datetime
import sys
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
