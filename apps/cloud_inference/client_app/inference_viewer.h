#ifndef INFERENCE_VIEWER_H
#define INFERENCE_VIEWER_H

#include "inference_receiver.h"
#include "perf_graph.h"
#include "perf_chart.h"
#include <QWidget>
#include <QFont>
#include <QThread>
#include <QVector>

namespace Ui {
class inference_viewer;
}

class inference_state {
public:
    inference_state();
    // configuration
    QFont statusBarFont;
    QFont buttonFont;
    // image data
    bool labelLoadDone;
    bool imageLoadDone;
    int imageLoadCount;
    bool imagePixmapDone;
    int imagePixmapCount;
    QVector<QByteArray> imageBuffer;
    QVector<QPixmap> imagePixmap;
    QVector<int> imageLabel;
    QVector<int> inferenceResultTop;
    QVector<QString> inferenceResultSummary;
    QVector<QString> shadowFileBuffer;
    // receiver
    QThread * receiver_thread;
    inference_receiver * receiver_worker;
    // rendering state
    float offsetSeconds;
    QVector<int> resultImageIndex;
    QVector<int> resultImageLabel;
    QVector<QString> resultImageSummary;
    QVector<QVector<int>> resultImageLabelTopK;
    QVector<QVector<float>> resultImageProbTopK;
    // mouse events
    bool abortLoadingRequested;
    bool exitButtonPressed;
    bool saveButtonPressed;
    QRect exitButtonRect;
    QRect saveButtonRect;
    QRect statusBarRect;
    bool mouseClicked;
    int mouseLeftClickX;
    int mouseLeftClickY;
    int mouseSelectedImage;
    bool viewRecentResults;
    QVector<QString> * dataLabels;
    QVector<QString> * dataHierarchy;
    int dataLabelCount;
    QString dataFilename;
    QString dataFolder;
    int imageDataSize;
    int imageDataStartOffset;
    QVector<QString> imageDataFilenames;
    int inputDim[3];
    int GPUs;
    int outputDim[3];
    QString serverHost;
    int serverPort;
    QString modelName;
    int maxImageDataSize;
    bool sendScaledImages;
    int sendFileName;
    int topKValue;
    //test summary
    int top1Count,top2Count,top3Count,top4Count,top5Count;
    int topKPassFail[100][2];
    int top5PassFail[100][10];
    int topKHierarchyPassFail[100][12];
    int topLabelMatch[1000][7];
    float top1TotProb,top2TotProb, top3TotProb, top4TotProb, top5TotProb;
    int totalMismatch, totalNoGroundTruth;
    float totalPassProb, totalFailProb;
    // performance results
    QRect perfButtonRect;
    bool perfButtonPressed;
    perf_graph performance;
    perf_chart chart;
    QString startTime;
    QElapsedTimer timerElapsed;
    QString elapsedTime;
    // performance graph
    QRect graphButtonRect;
    bool graphButtonPressed;
};

class inference_viewer : public QWidget
{
    Q_OBJECT

public:
    explicit inference_viewer(QString serverHost, int serverPort, QString modelName,
            QVector<QString> * dataLabels, QVector<QString> * dataHierarchy,
            QString dataFilename, QString dataFolder,
            int dimInput[3], int GPUs, int dimOutput[3], int maxImageDataSize,
            bool repeat_images, bool sendScaledImages, int enableSF, int topKValue,
            QWidget *parent = 0);
    ~inference_viewer();

public slots:
    void errorString(QString err);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void keyReleaseEvent(QKeyEvent *) override;

private:
    void startReceiver();
    void saveResults();
    void showPerfResults();
    void showChartResults();
    void terminate();

private:
    // ui
    Ui::inference_viewer *ui;
    QTimer * updateTimer;
    bool timerStopped;
    // state
    inference_state * state;
    QString fatalError;
    runtime_receiver_status progress;
};

/* Model Info Structure */
struct ModelMaster {
    QString name;
    int matched;
    int mismatched;
};
typedef struct ModelMaster ModelMasterInfo;

#endif // INFERENCE_VIEWER_H
