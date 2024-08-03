import sys
import math
from typing import Callable, Any 
from functools import partial
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QStackedWidget,
)
# Display:
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
# Window:
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480
WINDOW_TITLE = "Wikipedia Search Engine"
# Search box:
SEARCH_BOX_HEIGHT = 64
SEARCH_BOX_LABEL = "<h1>Wikipedia Search</h1>"
# Search button:
SEARCH_BUTTON_TEXT = "&Search"
# Data and results:
NUM_RESULTS_TO_SHOW = 2

class SearchWindow(QMainWindow):
    """
    Main window for search engine.
    """
    def __init__(self) -> None:
        super().__init__()

        # Widgets:
        self.generalWidget = QStackedWidget()

        # Getting layout for search page and adding to the generalWidget, 
        # retrieving search button widget for use in controller. The search 
        # page should always be at index 0 within the stacked widget:
        self.searchWidgets: dict[str, Any] = self.showSearchPage() 

        # Basic window formatting:
        self.setGeometry(
            math.floor(0.4 * DISPLAY_WIDTH),
            math.floor(0.4 * DISPLAY_HEIGHT),
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
        self.setWindowTitle(WINDOW_TITLE)

        # Set layout:
        self.setCentralWidget(self.generalWidget)

    def showSearchPage(self) -> dict[str, QWidget]:
        """
        Generates the layout for the search page, adds it to a widget instance,
        and adds that widget to the general (stacked) widget. 

        Returns a dictionary of each widget with it's variable name as key.
        """
        # Widgets:
        overallLayout = QVBoxLayout()
        searchBoxLabel = QLabel()
        searchBox = QLineEdit()
        searchButton = QPushButton(SEARCH_BUTTON_TEXT)

        # Formatting for widgets:
        searchBoxLabel.setText(SEARCH_BOX_LABEL)
        searchBox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        searchBox.setFixedHeight(SEARCH_BOX_HEIGHT)

        # Adding to layout:
        overallLayout.addWidget(searchBoxLabel)
        overallLayout.addWidget(searchBox)
        overallLayout.addWidget(searchButton)

        centralWidget = QWidget()
        centralWidget.setLayout(overallLayout)

        self.generalWidget.addWidget(centralWidget)

        return {
            "overallLayout": overallLayout,
            "searchBoxLabel": searchBoxLabel,
            "searchBox": searchBox,
            "searchButton": searchButton,
            "centralWidget": centralWidget
        }

    def showResultsPage(
        self,
        results: list[dict[str, str]]
        # num_results_to_show: int = NUM_RESULTS_TO_SHOW
        ) -> dict[str, Any]:
        """
        Shows results page for given results.
        """
        # List of keys within the results which we want to see:
        fields_to_show: tuple[str, ...] = ("name", "url")

        # Widgets:
        overallLayout = QVBoxLayout()
        # Adding results to widget, only showing NUM_RESULTS_TO_SHOW results:
        for result_idx in range(NUM_RESULTS_TO_SHOW):
            # Only showing desired fields. Here we construct the string which
            # will be the label, and then add it to the layout when it is fully
            # constructed:
            current_result_string_to_show: str = ""
            field: str
            for field in fields_to_show:
                # Inserting url as hyperlink:
                if field == "url":
                    current_result_string_to_show += (
                        f"<a href='{results[result_idx][field]}'>" + 
                        f"{results[result_idx][field]}</a>"
                    )
                else:
                    current_result_string_to_show += (
                        f"{results[result_idx][field]}\n"
                    )

            # Creating QLabel with string to show and adding to layout:
            resultLabelWidget = QLabel(current_result_string_to_show)
            resultLabelWidget.setTextFormat(Qt.RichText)
            resultLabelWidget.setTextInteractionFlags(
                Qt.LinksAccessibleByMouse
            )
            resultLabelWidget.setOpenExternalLinks(True)

            overallLayout.addWidget(resultLabelWidget)

        # Creating central widget:
        centralWidget = QWidget()
        centralWidget.setLayout(overallLayout)

        # Adding to general stacked widget of class instance. Placing at index
        # 1:
        self.generalWidget.insertWidget(1, centralWidget)

        # Switching general stacked widget's index to display results:
        self.generalWidget.setCurrentIndex(1)

        return {
            "overallLayout": overallLayout
        }

class SearchController:
    """
    Controller class for the search engine app window.
    """
    def __init__(self, model: Callable, view: SearchWindow, *model_args):
        # Attributes. model should be a function returning results represented
        # as list[dict[str, str]], a list of deserialised JSON objects:
        self._searchFunction: Callable = model
        self._view: SearchWindow = view
        # Arguments to be passed to model:
        self.model_args = model_args

        # Connect widgets:
        self._connectSignalsAndSlots()

    def _connectSignalsAndSlots(self) -> None:
        """
        Connects search button to given search function.
        """
        # Connecting search button to search function. showResultsPage takes
        # the results of the search function as it's argument:
        self._view.searchWidgets["searchButton"].clicked.connect(
            partial(
                self._view.showResultsPage, 
                self._searchFunction(*self.model_args)
            )
        )

def main() -> None:
    # Initialise app:
    searchApp = QApplication([])
    # Initialise window:
    searchWindow = SearchWindow()

    # Creating controller. Does not need to be stored as a variable as it holds
    # references to the model and view:
    def testSearchFunction(this: str):
        print(this)
        return [
            {"name": "blah",      "url": "here.com"},
            {"name": "more blah", "url": "there.org"}
        ]
    SearchController(testSearchFunction, searchWindow, "test")

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

if __name__ == "__main__":
    main()
