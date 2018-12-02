# from backend import Program
from program import Program
import tkinter as tk
import sys
from PyQt5.QtWidgets import QApplication

import time


if __name__ == "__main__":
    app = QApplication(sys.argv)
    p = Program()
    sys.exit(app.exec_())
