#ifndef INFERENCE_COMPILER_H
#define INFERENCE_COMPILER_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <mutex>

struct inference_compiler_status {
    bool completed;
    int errorCode;
    int modelFile1UploadProgress;
    int modelFile2UploadProgress;
    int compilationProgress;
    int dimOutput[3];
    QString message;
};

class inference_model_uploader : public QObject
{
    Q_OBJECT
public:
    explicit inference_model_uploader(
            bool enableServer,
            QString serverHost, int serverPort,
            int c, int h, int w,
            QString modelFile1, QString modelFile2,
            int reverseInputChannelOrder,
            float preprocessMpy[3],
            float preprocessAdd[3],
            QString compilerOptions,
            inference_compiler_status * progress,
            QObject *parent = nullptr);
    ~inference_model_uploader();

    static void abort();

signals:
    void finished();
    void error(QString err);

public slots:
    void run();

private:
    static bool abortRequested;

private:
    std::mutex mutex;
    // config
    bool enableServer;
    QString serverHost;
    int serverPort;
    int dimC;
    int dimH;
    int dimW;
    QString modelFile1;
    QString modelFile2;
    int reverseInputChannelOrder;
    float preprocessMpy[3];
    float preprocessAdd[3];
    QString compilerOptions;
    inference_compiler_status * progress;
};

class inference_compiler : public QWidget
{
    Q_OBJECT
public:
    explicit inference_compiler(
            bool enableServer,
            QString serverHost, int serverPort,
            int c, int h, int w,
            QString modelFile1, QString modelFile2,
            int reverseInputChannelOrder,
            float preprocessMpy[3],
            float preprocessAdd[3],
            QString compilerOptions,
            inference_compiler_status * progress,
            QWidget *parent = nullptr);

protected:
    void Ok();
    void Cancel();
    void startModelUploader();

signals:

public slots:
    void tick();
    void errorString(QString err);

private:
    // config
    bool enableServer;
    QString serverHost;
    int serverPort;
    int dimC;
    int dimH;
    int dimW;
    QString modelFile1;
    QString modelFile2;
    int reverseInputChannelOrder;
    float preprocessMpy[3];
    float preprocessAdd[3];
    QString compilerOptions;
    // status
    QLabel * labelStatus;
    QLineEdit * editModelFile1UploadProgress;
    QLineEdit * editModelFile2UploadProgress;
    QLineEdit * editCompilerProgress;
    QLineEdit * editDimC;
    QLineEdit * editDimH;
    QLineEdit * editDimW;
    QLineEdit * editOutDimC;
    QLineEdit * editOutDimH;
    QLineEdit * editOutDimW;
    QLineEdit * editCompilerMessage;
    QPushButton * okCompilerButton;
    QPushButton * cancelCompilerButton;
    inference_compiler_status * progress;
    inference_model_uploader * worker;
};

#endif // INFERENCE_COMPILER_H
