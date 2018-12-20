#ifndef INFERENCE_CONTROL_H
#define INFERENCE_CONTROL_H

#include "inference_viewer.h"
#include "infcom.h"
#include "inference_compiler.h"
#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>

class inference_control : public QWidget
{
    Q_OBJECT
public:
    explicit inference_control(int operationMode, QWidget *parent = nullptr);

signals:

public slots:
    void tick();
    void runConnection();
    void runCompiler();
    void runInference();
    void modelSelect(int model);
    void browseModelFile1();
    void browseModelFile2();
    void exitControl();
    void onChangeDimH(const QString &);
    void onChangeDimW(const QString &);
    void onChangeModelFile1(const QString &);
    void onChangeModelFile2(const QString &);
    void onChangeInputInverserOrder(int order);
    void onChangePreprocessMpyC0(const QString &);
    void onChangePreprocessMpyC1(const QString &);
    void onChangePreprocessMpyC2(const QString &);
    void onChangePreprocessAddC0(const QString &);
    void onChangePreprocessAddC1(const QString &);
    void onChangePreprocessAddC2(const QString &);
    void onChangePublishMode(int mode);
    void onChangeModelName(const QString &);
    void onLogo1Click();
    void onLogo2Click();
    void topKResultsEnable(bool topKEnable);
    void shadowFolderEnable(bool shadowEnable);

protected:
    void browseShadowFolder();
    void browseDataLabels();
    void browseDataHierarchy();
    void browseDataFilename();
    void browseDataFolder();
    bool isConfigValid(QPushButton * button, QString& err);
    void saveConfig();
    void loadConfig();

private:
    struct ModelInfo {
        QString name;
        int inputDim[3];
        int outputDim[3];
        int reverseInputChannelOrder;
        float preprocessMpy[3];
        float preprocessAdd[3];
    };
    QLineEdit * editServerHost;
    QLineEdit * editServerPort;
    QLineEdit * editServerPassword;
    QPushButton * buttonConnect;
    QLabel * labelServerStatus;
    QComboBox * comboModelSelect;
    QPushButton * buttonCompile;
    QLineEdit * editDimH;
    QLineEdit * editDimW;
    QLineEdit * editOutDimC;
    QLineEdit * editOutDimH;
    QLineEdit * editOutDimW;
    QLabel * labelModelFile1;
    QLabel * labelModelFile2;
    QLabel * labelCompilerOptions;
    QLineEdit * editModelFile1;
    QLineEdit * editModelFile2;
    QComboBox * comboInvertInputChannels;
    QComboBox * comboPublishOptions;
    QComboBox * comboTopKResult;
    QLineEdit * editModelName;
    QPushButton * buttonModelFile1;
    QPushButton * buttonModelFile2;
    QLabel * labelPreprocessMpy;
    QLineEdit * editPreprocessMpyC0;
    QLineEdit * editPreprocessMpyC1;
    QLineEdit * editPreprocessMpyC2;
    QLabel * labelPreprocessAdd;
    QLineEdit * editPreprocessAddC0;
    QLineEdit * editPreprocessAddC1;
    QLineEdit * editPreprocessAddC2;
    QLabel * labelCompilerStatus;
    QLineEdit * editGPUs;
    QLabel * labelMaxGPUs;
    QPushButton * buttonInference;
    QLineEdit * editImageLabelsFile;
    QLineEdit * editImageHierarchyFile;
    QLineEdit * editImageFolder;
    QLineEdit * editImageListFile;
    QLineEdit * editMaxDataSize;
    QCheckBox * checkRepeatImages;
    QCheckBox * checkScaledImages;
    QCheckBox * checkTopKResult;
    QCheckBox * checkShadowFolder;
    QLineEdit * editShadowFolderAddr;
    QPushButton * buttonShadowFolder;
    inference_compiler_status compiler_status;
    bool operationMode;
    bool connectionSuccessful;
    int modelType;
    int numModelTypes;
    int maxGPUs;
    int enableSF;
    int sendFileName;
    int enableTopK;
    int topKValue;
    QVector<QString> * dataLabels;
    QVector<QString> * dataHierarchy;
    QVector<QString> typeModelFile1Label;
    QVector<QString> typeModelFile1Desc;
    QVector<QString> typeModelFile2Label;
    QVector<QString> typeModelFile2Desc;
    QVector<ModelInfo> modelList;
    QString lastModelFile1;
    QString lastModelFile2;
    int lastInverseInputChannelOrder;
    int lastPublishMode;
    QString lastModelName;
    QString lastDimW;
    QString lastDimH;
    QString lastPreprocessMpy[3];
    QString lastPreprocessAdd[3];
};

#endif // INFERENCE_CONTROL_H
