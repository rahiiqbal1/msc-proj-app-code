import sys
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
    QScrollArea,
)
from PyQt5.QtGui import QFont

# Window:
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 480
WINDOW_TITLE = "Wikipedia Search Engine"
# All-widgets:
FONT_TO_USE = "Times New Roman"
# Search box label:
SEARCH_BOX_LABEL_FONT_SIZE = 22
# Search box:
SEARCH_BOX_HEIGHT = 64
SEARCH_BOX_WIDTH = 1024
SEARCH_BOX_LABEL = "<h1>Wikipedia Search</h1>"
SEARCH_BOX_FONT_SIZE = 18
# Search button:
SEARCH_BUTTON_TEXT = "Search"
SEARCH_BUTTON_FONT_SIZE = 18
# Search again button:
SEARCH_AGAIN_BUTTON_TEXT = "Search"
SEARCH_AGAIN_BUTTON_FONT_SIZE = 18
# General font size for results text:
RESULT_GENERAL_FONT_SIZE = 18
# Results name:
RESULT_NAME_FONT_SIZE = 22
# Results url:
RESULT_URL_FONT_SIZE = 18
# Results scrollbar:
RESULTS_SCROLL_AREA_HEIGHT = 2500
# Data and results:
NUM_RESULTS_TO_SHOW = 10
INTRA_RESULT_SPACING = 25

def main() -> None:
    # Initialise app:
    searchApp = QApplication([])
    # Initialise window:
    searchWindow = SearchWindow()

    # Creating controller. Does not need to be stored as a variable as it holds
    # references to the model and view:
    def testSearchFunction(_: str) -> list[dict[str, str]]:
        return [
            {"name": "blah",      "abstract": "cool stuff", "url": "here.com"},
            {"name": "more blah", "abstract": "very nice",  "url": "there.org"}
        ] * (NUM_RESULTS_TO_SHOW // 2)
    SearchController(testSearchFunction, searchWindow)

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

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
        self.searchWidgets: dict[str, Any] = self._showSearchPage() 
        # Adding search again button to widgets for class:
        self.searchAgainButton = QPushButton(SEARCH_AGAIN_BUTTON_TEXT)
        self.searchAgainButton.setShortcut("Return")
        # Adding results page search box to widgets for class:
        self.rpSearchBox = QLineEdit()
        self.rpSearchBox.setFixedHeight(SEARCH_BOX_HEIGHT)
        self.rpSearchBox.setFont(QFont(FONT_TO_USE, SEARCH_BOX_FONT_SIZE))

        # Basic window formatting:
        self.showMaximized()
        self.setWindowTitle(WINDOW_TITLE)

        # Set layout:
        self.setCentralWidget(self.generalWidget)

    def _showSearchPage(self) -> dict[str, QWidget]:
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
        searchBoxLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        searchBoxLabel.setFont(QFont(FONT_TO_USE, SEARCH_BOX_LABEL_FONT_SIZE))

        searchBox.setFixedWidth(SEARCH_BOX_WIDTH)
        searchBox.setFixedHeight(SEARCH_BOX_HEIGHT)
        searchBox.setFont(QFont(FONT_TO_USE, SEARCH_BOX_FONT_SIZE))

        searchButton.setFixedWidth(SEARCH_BOX_WIDTH)
        searchButton.setFixedHeight(SEARCH_BOX_HEIGHT)
        searchButton.setFont((QFont(FONT_TO_USE, SEARCH_BUTTON_FONT_SIZE)))
        searchButton.setShortcut("Return")

        overallLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Adding to layout:
        overallLayout.addStretch()
        overallLayout.addWidget(searchBoxLabel)
        overallLayout.addWidget(searchBox)
        overallLayout.addWidget(searchButton)
        overallLayout.addStretch()

        centralWidget = QWidget()
        centralWidget.setLayout(overallLayout)

        self.generalWidget.insertWidget(0, centralWidget)

        return {
            "overallLayout": overallLayout,
            "searchBoxLabel": searchBoxLabel,
            "searchBox": searchBox,
            "searchButton": searchButton,
            "centralWidget": centralWidget
        }
    
    def switchToSearchPage(self) -> None:
        """
        Switches to the search page, given that it's index is 0 in the stacked
        widget that is the generalWidget. Clears the search bar before 
        switching.
        """
        self.searchWidgets["searchBox"].setText("")
        self.generalWidget.setCurrentIndex(0)

    def showResultsPage(self, searchFunction: Callable) -> dict[str, Any]:
        """
        Shows results page using given search function and query.
        """
        # List of keys within the results which we want to see:
        fieldsToShow: tuple[str, ...] = ("name", "abstract", "url")

        # Evaluating results. The searchFunction here takes no arguments as
        # this method is connected to the searchFunction within the 
        # SearchController, which searches using the text present in the 
        # search box on the search page:
        results: list[dict[str, str]] = searchFunction()

        # Getting VBoxLayout containing all results:
        overallVBoxLayout: QVBoxLayout = (
            self._getAllResultsVBoxLayout(results, fieldsToShow)
        )
        # Adding search box to top of layout:
        overallVBoxLayout.insertWidget(0, self.rpSearchBox)

        # Creating scroll layout, using a subwidget to be able to add to the
        # scroll area:
        scrollArea = QScrollArea()
        subWidget = QWidget()
        subWidget.setLayout(overallVBoxLayout)
        scrollArea.setWidget(subWidget)
        scrollArea.setWidgetResizable(True)

        # Adding to general stacked widget of class instance. Placing at index
        # 1:
        self.generalWidget.insertWidget(1, scrollArea)

        # Switching general stacked widget's index to display results:
        self.generalWidget.setCurrentIndex(1)

        return {
            "scrollArea": scrollArea
        }

    def _getAllResultsVBoxLayout(
        self,
        results: list[dict[str, str]],
        fieldsToShow: tuple[str, ...]
        ) -> QVBoxLayout:
        """
        Returns a QVBoxLayout of the list of results, showing only the
        specified fields.
        """
        # Layout and widgets:
        overallVBoxLayout = QVBoxLayout()
        self.searchAgainButton.setFont(
            QFont(FONT_TO_USE, SEARCH_AGAIN_BUTTON_FONT_SIZE)
        )

        # Adding search again button to overall layout:
        overallVBoxLayout.addWidget(self.searchAgainButton)

        # Adding results to widget, only showing a maximum of
        # NUM_RESULTS_TO_SHOW results, although there may be less than this for
        # some queries:
        resultIdx: int
        singleResult: dict[str, str]
        for resultIdx, singleResult in enumerate(results):
            # Leave if we have already reached the required number of results
            # to show (in the previous iteration):
            if resultIdx == NUM_RESULTS_TO_SHOW:
                break

            # Getting QVBoxLayout for this result containing QLabels with each
            # desired field as text:
            thisResultVBoxLayout: QVBoxLayout = (
                self._getSingleResultVBoxLayout(singleResult, fieldsToShow)
            )

            # Adding result layout to the overall layout. Uses a sub-widget to
            # make this possible:
            subWidget = QWidget()
            subWidget.setLayout(thisResultVBoxLayout)
            overallVBoxLayout.addWidget(subWidget)

        return overallVBoxLayout

    def _getSingleResultVBoxLayout(
        self,
        singleResult: dict[str, str],
        fieldsToShow: tuple[str, ...]
        ) -> QVBoxLayout:
        """ Gets the desired text fields from the result given and returns them
        as a QVBoxLayout object with each field it's own QLabel within the
        layout.
        """
        # Initialising VBoxLayout which will be returned:
        thisResultVBoxLayout = QVBoxLayout()
        thisResultVBoxLayout.setSpacing(INTRA_RESULT_SPACING)

        # Only showing desired fields:
        field: str
        for field in fieldsToShow:
            # Initialising string which will be added to a QLabel object and
            # then to the layout for the current result:
            currentFieldStringToShow: str = ""

            if field == "name":
                try: 
                    currentFieldStringToShow += singleResult[field]
                except KeyError:
                    currentFieldStringToShow += (f"No '{field}' found.")

                # For the name (i.e. title) set the font size to the chosen
                # size:
                thisFieldLabelWidget = QLabel(currentFieldStringToShow)
                thisFieldLabelWidget.setFont(
                    QFont(
                        FONT_TO_USE, RESULT_NAME_FONT_SIZE, weight = QFont.Bold
                    )
                )

            elif field == "url":
                try:
                    # Inserting url as hyperlink:
                    currentFieldStringToShow += (
                        f"<a href='{singleResult[field]}'>" + 
                        f"{singleResult[field]}</a>"
                    )
                except KeyError:
                    currentFieldStringToShow += (f"No '{field}' found.")

                # Creating Qlabel widget with the string for the current field
                # as the text:
                thisFieldLabelWidget = QLabel(currentFieldStringToShow)
                thisFieldLabelWidget.setTextInteractionFlags(
                    Qt.LinksAccessibleByMouse
                )
                thisFieldLabelWidget.setOpenExternalLinks(True)
                thisFieldLabelWidget.setFont(
                    QFont(FONT_TO_USE, RESULT_URL_FONT_SIZE)
                )

            else:
                # Showing only a fixed amount of the text in the field:
                field_len_to_show: int = 100
                # Setting string to the text of the field:
                try:
                    currentFieldStringToShow += (
                        singleResult[field][: field_len_to_show] + "..."
                    )
                except KeyError:
                    currentFieldStringToShow += (f"No '{field}' found.")

                # Creating a label with the string as text:
                thisFieldLabelWidget = QLabel(currentFieldStringToShow)
                thisFieldLabelWidget.setFont(
                    QFont(FONT_TO_USE, RESULT_GENERAL_FONT_SIZE)
                )

            # Adding the current field's label to the result's layout:
            thisResultVBoxLayout.addWidget(thisFieldLabelWidget)

        return thisResultVBoxLayout

class SearchController:
    """
    Controller class for the search engine app window.
    """
    def __init__(self, model: Callable, view: SearchWindow, *model_args):
        # Attributes. model should be a function returning results represented
        # as list[dict[str, str]], a list of deserialised JSON objects. The 
        # final argument of model must be the search query:
        self._model: Callable = model
        self._view: SearchWindow = view
        # Arguments to be passed to model, to be unpacked when used:
        self._modelArgs: tuple[Any, ...] = model_args

        # Connect widgets:
        self._connectSignalsAndSlots()

    def _searchFunction(self) -> list[dict[str, str]]:
        """
        Deals with searching through data with query present in search box.
        """
        # If there is text in the search box on the home page, use that for the
        # search. Otherwise, use the text in the box on the results page. This
        # allows the box on the results page to be used for searches, but 
        # relies on the search box on the home page being cleared after use:
        searchPageQuery: str = self._view.searchWidgets["searchBox"].text()
        if  searchPageQuery != "":
            self._view.rpSearchBox.setText(searchPageQuery)
            searchQueryUsed: str = self._view.searchWidgets["searchBox"].text()
        
        else:
            searchQueryUsed: str = self._view.rpSearchBox.text()

        # Setting results page search box text to that of the home page search
        # bar, then setting home page search box text blank after use:
        self._view.searchWidgets["searchBox"].setText("")

        # Passing given modelArgs to model along with search query present in
        # text box:
        return self._model(*(self._modelArgs + (searchQueryUsed, )))

    def _connectSignalsAndSlots(self) -> None:
        """
        Connects search button to given search function.
        """
        # Connecting search button to search function. showResultsPage takes
        # the results of the search function as it's argument:
        self._view.searchWidgets["searchButton"].clicked.connect(
            partial(
                self._view.showResultsPage, 
                self._searchFunction
            )
        )

        # Connecting search again button to a lambda function that switches 
        # back to the search page by it's index in the general(stacked) widget:
        self._view.searchAgainButton.clicked.connect(
            partial(
                self._view.showResultsPage, 
                self._searchFunction
            )
        )

if __name__ == "__main__":
    main()
