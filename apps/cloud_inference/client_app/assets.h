#ifndef ASSETS_H
#define ASSETS_H


class assets
{
public:
    static const char * getLogoPng1Buf() { return (const char *) logoPng1Buf; }
    static const char * getLogoPng2Buf() { return (const char *) logoPng2Buf; }
    static int getLogoPng1Len() { return 3686; }
    static int getLogoPng2Len() { return 5311; }

private:
    static unsigned int logoPng1Buf[];
    static unsigned int logoPng2Buf[];
};

#endif // ASSETS_H
