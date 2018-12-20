#include "inference_panel.h"
#include "ui_inference_panel.h"

inference_panel::inference_panel(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::inference_panel)
{
    ui->setupUi(this);
    display_panel.setWindowIcon(QIcon(":/images/vega_icon_150.png"));
    connect(ui->viewGraph_pushButton, &QAbstractButton::clicked, this, &inference_panel::viewPerformanceGraph);
}

inference_panel::~inference_panel()
{
    delete ui;
}

void inference_panel::viewPerformanceGraph()
{
    display_panel.show();
}


