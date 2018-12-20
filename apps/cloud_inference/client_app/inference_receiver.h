#ifndef INFERENCE_RECEIVER_H
#define INFERENCE_RECEIVER_H

#include <QObject>
#include <QVector>
#include <QQueue>
#include <QMutex>
#include <QElapsedTimer>
#include <QMouseEvent>
#include <mutex>

struct runtime_receiver_status {
    bool completed;
    int errorCode;
    QString message;
    bool repeat_images;
    bool completed_send;
    bool completed_decode;
    bool completed_load;
    int images_loaded;
    int images_decoded;
    int images_sent;
    int images_received;
};

class inference_receiver : public QObject
{
    Q_OBJECT
public:
    explicit inference_receiver(
            QString serverHost, int serverPort, QString modelName,
            int GPUs, int * inputDim, int * outputDim, const char * runtimeOptions,
            QVector<QByteArray> * imageBuffer,
            runtime_receiver_status * progress, int sendFileName, int topKValue,
            QVector<QString> * shadowFileBuffer,
            QObject *parent = nullptr);
    ~inference_receiver();

    static void abort();
    void setImageCount(int imageCount, int labelCount, QVector<QString> * dataLabels);
    void getReceivedList(QVector<int>& indexQ, QVector<int>& labelQ, QVector<QString>& summaryQ,
                         QVector<QVector<int> >& labelTopK, QVector<QVector<float> >& probTopK);
    float getPerfImagesPerSecond();

signals:
    void finished();
    void error(QString err);

public slots:
    void run();

private:
    static bool abortRequsted;

private:
    std::mutex mutex;
    int imageCount;
    int labelCount;
    QQueue<int> imageIndex;
    QQueue<int> imageLabel;
    QQueue<QString> imageSummary;
    QQueue<QVector<int>> imageTopkLabels;
    QQueue<QVector<float>> imageTopkConfidence;
    QElapsedTimer perfTimer;
    int perfImageCount;
    float perfRate;
    QVector<QByteArray> * imageBuffer;
    QVector<QString> * dataLabels;
    QVector<QString> * shadowFileBuffer;
    QString serverHost;
    int serverPort;
    QString modelName;
    int GPUs;
    int * inputDim;
    int * outputDim;
    const char * runtimeOptions;
    runtime_receiver_status * progress;
    int sendFileName;
    int topKValue;
};

#endif // INFERENCE_RECEIVER_H
