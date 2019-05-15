#include "perf_chart.h"
#include "ui_perf_chart.h"

perf_chart::perf_chart(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::perf_chart)
{
    ui->setupUi(this);
    mFPSValue = 0;
    mMaxFPS = 0;
    initGraph();
}

perf_chart::~perf_chart()
{
    delete ui;
}

void perf_chart::initGraph()
{
    ui->CustomPlot->addGraph();

    QSharedPointer<QCPAxisTickerTime> timeTicker(new QCPAxisTickerTime);
    timeTicker->setTimeFormat("%h:%m:%s");

    // set default time duration to rb1 value
    ui->rb1->setChecked(1);

    // set default option to moving moving graph
    ui->movingGraph->setChecked(1);

    // x axis
    ui->CustomPlot->xAxis->setTicker(timeTicker);
    ui->CustomPlot->xAxis->setTickLabelFont(QFont(QFont().family(), 12));
    ui->CustomPlot->xAxis->setLabel("Time Elapsed");
    ui->CustomPlot->xAxis->setLabelFont((QFont(QFont().family(), 20)));

    // y axis
    ui->CustomPlot->yAxis->setTickLabelFont(QFont(QFont().family(), 12));
    ui->CustomPlot->yAxis->setLabel("FPS");
    ui->CustomPlot->yAxis->setLabelFont((QFont(QFont().family(), 20)));

    ui->CustomPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    ui->CustomPlot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignTop);

    // make left and bottom axes transfer their ranges to right and top axes:
    connect(ui->CustomPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), ui->CustomPlot->xAxis2, SLOT(setRange(QCPRange)));
    connect(ui->CustomPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), ui->CustomPlot->yAxis2, SLOT(setRange(QCPRange)));

    // setup a timer that repeatedly calls MainWindow::realtimeDataSlot:
    connect(&timer, SIGNAL(timeout()), this, SLOT(RealtimeDataSlot()));

    // rescale axis
    connect(ui->rb1, &QAbstractButton::clicked, this, &perf_chart::rescaleAxis);
    connect(ui->rb2, &QAbstractButton::clicked, this, &perf_chart::rescaleAxis);
    connect(ui->rb3, &QAbstractButton::clicked, this, &perf_chart::rescaleAxis);
    connect(ui->rb4, &QAbstractButton::clicked, this, &perf_chart::rescaleAxis);
    connect(ui->rb5, &QAbstractButton::clicked, this, &perf_chart::rescaleAxis);

    // close button
    connect(ui->closeButton, &QAbstractButton::clicked, this, &perf_chart::closeChartView);
    timer.start(5);
}

void perf_chart::RealtimeDataSlot()
{
    static QTime time(QTime::currentTime());
    double key = time.elapsed()/1000.0; // time elapsed since start of demo, in seconds
    static double lastPointKey = 0;
    if (key-lastPointKey > 0.01) { // at most add point every 10 ms
        ui->CustomPlot->graph(0)->addData(key, mFPSValue);
        lastPointKey = key;
        if (mFPSValue > mMaxFPS) {
            mMaxFPS = mFPSValue;
        }
    }
    if (ui->movingGraph->isChecked()) {
        rescaleAxis(key);
    }
    ui->CustomPlot->replot();
}

void perf_chart::rescaleAxis(double key) {
    ui->CustomPlot->graph()->rescaleValueAxis();
    if (ui->rb1->isChecked()) {
        ui->CustomPlot->xAxis->setRange(key+20, 60, Qt::AlignRight);
        mRangeX = 60;
    } else if (ui->rb2->isChecked()) {
        ui->CustomPlot->xAxis->setRange(key+30, 120, Qt::AlignRight);
        mRangeX = 120;
    } else if (ui->rb3->isChecked()) {
        ui->CustomPlot->xAxis->setRange(key+30, 240, Qt::AlignRight);
        mRangeX = 240;
    } else if (ui->rb4->isChecked()) {
        ui->CustomPlot->xAxis->setRange(key+30, 480, Qt::AlignRight);
        mRangeX = 480;
    } else if (ui->rb5->isChecked()) {
        ui->CustomPlot->xAxis->setRange(0, key+100);
        mRangeX = key+100;
    }
    mRangeY = mMaxFPS*1.5;
    ui->CustomPlot->yAxis->setRange(0, mRangeY);
}

void perf_chart::updateFPSValue(int fpsValue)
{
    mFPSValue = fpsValue;
    if (mFPSValue > mMaxFPS)
        mMaxFPS = mFPSValue;
}

void perf_chart::setGPUs(int numGPUs)
{
    mNumGPUs = numGPUs;
    ui->gpus_lcdNumber->display(numGPUs);
}

void perf_chart::closeChartView()
{
    this->close();
}

