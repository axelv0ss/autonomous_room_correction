# from backend import Program
from gui import GUI
import tkinter as tk
import sys
from PyQt5.QtWidgets import QApplication

import time


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())
