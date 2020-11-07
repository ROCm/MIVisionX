import os
from PyQt4 import QtGui, uic

import adat_classification as ac


class inference_control(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(inference_control, self).__init__(parent)
        self.configDict = {}
        self.runningState = False
        self.lastGeneratedFile = None
        self.initUI()

    def initUI(self):
        uic.loadUi("adatui.ui", self)
        # self.setStyleSheet("background-color: white")
        self.btnGenerate.setEnabled(False)
        self.btnOpenHtml.setEnabled(False)
        self.btnGenerate.setStyleSheet("background-color: 0")

        # handle menu buttons
        self.actionOpen_Config.triggered.connect(self.loadConfig)
        self.actionSave_Config.triggered.connect(self.saveConfig)

        self.btnOpenConfig.clicked.connect(self.loadConfig)
        self.btnSaveConfig.clicked.connect(self.saveConfig)
        self.btnOpenHtml.clicked.connect(self.openHtml)

        self.btnFileBrowse.clicked.connect(self.browseFile)
        self.btnOutputBrowse.clicked.connect(self.browseOutput)
        self.btnLabelBrowse.clicked.connect(self.browseLabel)
        self.btnImageDirBrowse.clicked.connect(self.browseImage)
        self.btnHierarchyFileBrowse.clicked.connect(self.browseHier)

        self.btnGenerate.clicked.connect(self.runGenerate)

        self.txtInputFile.textChanged.connect(self.checkInput)
        self.txtOutputDir.textChanged.connect(self.checkInput)
        self.txtLabelFile.textChanged.connect(self.checkInput)
        self.txtOutputName.textChanged.connect(self.checkInput)

        self.txtInputFile.setPlaceholderText("[required]")
        self.txtOutputDir.setPlaceholderText("[required]")
        self.txtLabelFile.setPlaceholderText("[required]")
        self.txtImageDir.setPlaceholderText("[optional]")
        self.txtHierarchyFile.setPlaceholderText("[optional]")
        self.txtOutputName.setPlaceholderText("[optional]")
        self.txtModelName.setPlaceholderText("[optional]")

        # self.close_pushButton.setStyleSheet(
        #     "color: white; background-color: darkRed")
        # self.readSetupFile()
        self.show()

    def browseFile(self):
        self.txtInputFile.setText(
            QtGui.QFileDialog.getOpenFileName(self, 'Open File', './', '*.csv'))

    def browseOutput(self):
        self.txtOutputDir.setText(
            QtGui.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseLabel(self):
        self.txtLabelFile.setText(
            QtGui.QFileDialog.getOpenFileName(self, 'Open File', './', '*.txt'))

    def browseImage(self):
        self.txtImageDir.setText(
            QtGui.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseVal(self):
        self.val_lineEdit.setText(QtGui.QFileDialog.getOpenFileName(
            self, 'Open File', './', '*.txt'))

    def browseHier(self):
        self.txtHierarchyFile.setText(QtGui.QFileDialog.getOpenFileName(
            self, 'Open File', './', '*.csv'))

    def setConfigDict(self, configDict):
        self.configDict = configDict
        self.txtModelName.setText(self.configDict.get('model_name', ''))
        self.txtInputFile.setText(self.configDict.get('inference_results', ''))    
        self.txtLabelFile.setText(self.configDict.get('label', ''))        
        self.txtHierarchyFile.setText(self.configDict.get('hierarchy', ''))        
        self.txtOutputDir.setText(self.configDict.get('output_dir', ''))        
        self.txtImageDir.setText(self.configDict.get('image_dir', ''))        
        self.txtOutputName.setText(self.configDict.get('output_name', ''))

    def loadConfig(self):
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open File', './', 'Config Files (*.json *.ini)')

        if str(filename) != '':
            configDict = ac.readConfig(str(filename), None)
            self.setConfigDict(configDict)
            self.statusbar.showMessage('Config loaded..')

    def saveConfig(self):
        self.getConfig()
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save Config', './', 'INI File (*.ini);; JSON File (*.json)')

        if str(filename) != '':
            ac.saveConfig(str(filename), self.configDict)
            self.statusbar.showMessage('Config Saved..')

    def checkInput(self):
        if not str(self.txtInputFile.text()) == '' and not str(self.txtLabelFile.text()) == '' \
                and not str(self.txtOutputDir.text()) == '' \
                and not str(self.txtOutputName.text()) == '':
            self.btnGenerate.setEnabled(True)
            self.btnGenerate.setStyleSheet("background-color: lightgreen")
        else:
            self.btnGenerate.setEnabled(False)
            self.btnGenerate.setStyleSheet("background-color: 0")

    def getVal(self, obj):
        return str(obj) if str(obj) else None

    def showErrorDialog(self, err):
        msg = QtGui.QMessageBox()
        msg.setIcon(QtGui.QMessageBox.Critical)
        msg.setText("One or more config values are not valid:")
        msg.setInformativeText(str(err))
        msg.setWindowTitle("Error")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QtGui.QMessageBox.Ok)

        msg.exec_()

    def getConfig(self):
        self.configDict['model_name'] = self.getVal(self.txtModelName.text())
        self.configDict['inference_results'] = self.getVal(
            self.txtInputFile.text())
        self.configDict['label'] = self.getVal(self.txtLabelFile.text())
        self.configDict['hierarchy'] = self.getVal(
            self.txtHierarchyFile.text())
        self.configDict['output_dir'] = self.getVal(self.txtOutputDir.text())
        self.configDict['image_dir'] = self.getVal(self.txtImageDir.text())
        self.configDict['output_name'] = self.getVal(
            self.txtOutputName.text())

    def runGenerate(self):
        self.getConfig()
        try:
            ac.validateConfig(self.configDict)
        except Exception as ex:
            self.showErrorDialog(ex)
            return

        ac.generateAnalysisOutput(self.configDict)
        self.statusbar.showMessage('Html Output Generated!')
        self.lastGeneratedFile = os.path.join(
            self.configDict['output_dir'], self.configDict['output_name']+'-toolKit', 'index.html')

        self.btnOpenHtml.setEnabled(True)

    def openHtml(self):
        import webbrowser
        webbrowser.open(self.lastGeneratedFile)
        # os.system('start ' + self.lastGeneratedFile)

    def closeEvent(self, event):
        if not self.runningState:
            exit(0)


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    mainWindow = inference_control()
    mainWindow.show()
    sys.exit(app.exec_())
