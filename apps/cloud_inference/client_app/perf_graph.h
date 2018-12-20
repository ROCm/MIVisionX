#ifndef PERF_GRAPH_H
#define PERF_GRAPH_H

#include <QDialog>

namespace Ui {
class perf_graph;
}

class perf_graph : public QDialog
{
    Q_OBJECT

public:
    explicit perf_graph(QWidget *parent = 0);
    ~perf_graph();

private:
    Ui::perf_graph *ui;
    float maxFPS;

public slots:
     void closePerformanceView();
     void resetPerformanceView();
     void setModelName(QString ModelName);
     void setNumGPU(int numGPU);
     void setStartTime(QString startTime);
     void updateElapsedTime(QString elapsedTime);
     void updateFPSValue(float fps);
     void updateTotalImagesValue(int images);
};

#endif // PERF_GRAPH_H
