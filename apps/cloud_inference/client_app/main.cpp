#include "inference_control.h"
#include <QApplication>
#include <QSplashScreen>
#include <QTime>
#include <QIcon>

void splashDelay(int millisecondsToWait)
{
    QTime dieTime = QTime::currentTime().addMSecs(millisecondsToWait);
    while( QTime::currentTime() < dieTime )
    {
        QCoreApplication::processEvents( QEventLoop::AllEvents, 100 );
    }
}

int main(int argc, char *argv[])
{
    int enable_repeat_images = 1;
    if(argv[1]) enable_repeat_images = atoi(argv[1]);
    QApplication a(argc, argv);
    inference_control control(enable_repeat_images);
    QSplashScreen splash;
    splash.setPixmap(QPixmap(":/images/inference_app_splash.png"));
    splash.show(); splashDelay(2000); splash.hide();
    control.setWindowIcon(QIcon(":/images/vega_icon_150.png"));
    control.show();

    return a.exec();
}
