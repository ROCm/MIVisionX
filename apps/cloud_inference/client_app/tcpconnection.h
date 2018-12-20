#ifndef TCPCONNECTION_H
#define TCPCONNECTION_H

#include "infcom.h"
#include <QTcpSocket>

class TcpConnection : public QTcpSocket
{
public:
    TcpConnection(QString hostname, int port, int timeout, QObject *parent = Q_NULLPTR);
    bool connected();

    bool recvCmd(InfComCommand& cmd);
    bool sendCmd(const InfComCommand& cmd);
    bool sendFile(int command, const QString fileName, volatile int& progress, QString& mesg, volatile bool& abortRequested);
    bool sendImage(int tag, QByteArray& byteArray, int& errorCode, QString& message, volatile bool& abortRequested);

protected:
    int recv(void * pkt, size_t len);
    int send(const void * pkt, size_t len);

private:
    int error;
    int timeout;
};

#endif // TCPCONNECTION_H
