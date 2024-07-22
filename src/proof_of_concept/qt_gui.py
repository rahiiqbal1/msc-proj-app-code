import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QLineEdit,
    QPushButton,
)

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480

class SearchWindow(QMainWindow):
    """
    Main window for search engine.
    """
    def __init__(self, title: str):
        super().__init__()
        # Attributes:
        self.generalLayout = QGridLayout()

        centralWidget = QWidget(self)
        centralWidget.setLayout(self.generalLayout)

        # Setting window format and widgets:
        self.setWindowTitle(title)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setCentralWidget(centralWidget)


def main() -> None:
    # Initialise app:
    searchApp = QApplication([])
    # Initialise window:
    searchWindow = SearchWindow("test")

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

if __name__ == "__main__":
    main()
