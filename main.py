from program import Program
import sys
from PyQt5.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    p = Program()
    sys.exit(app.exec_())
