#ifndef INFERENCE_PANEL_H
#define INFERENCE_PANEL_H

#include <QDialog>
#include "perf_graph.h"

namespace Ui {
class inference_panel;
}

class inference_panel : public QDialog
{
    Q_OBJECT

public:
    explicit inference_panel(QWidget *parent = 0);
    ~inference_panel();

private:
    Ui::inference_panel *ui;
    perf_graph display_panel;

public slots:
     void viewPerformanceGraph();
};

#endif // INFERENCE_PANEL_H
