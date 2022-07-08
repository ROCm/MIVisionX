#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include "profiler.h"
#include "common.h"
#include <iostream>
#if PROFILER_MODE

#if _WIN32
#include <windows.h>
#else
#include <x86intrin.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#endif
#include <stdio.h>

#define PROFILER_DEFINE_EVENT_NAME_ENUM(g,e) #g "-" #e,
const char * ProfilerEventName[] = {
    PROFILER_DEFINE_EVENT_NAME_ENUM(inference_server_app, workMasterInputQ)
    PROFILER_DEFINE_EVENT_NAME_ENUM(inference_server_app, workDeviceInputCopyBatch)
    PROFILER_DEFINE_EVENT_NAME_ENUM(inference_server_app, workDeviceInputCopyJpegDecode)
    PROFILER_DEFINE_EVENT_NAME_ENUM(inference_server_app, workDeviceProcess)
    PROFILER_DEFINE_EVENT_NAME_ENUM(inference_server_app, workDeviceOutputCopy)
    PROFILER_DEFINE_EVENT_NAME_ENUM(inference_server_app, workRGBtoTensor)
    ""
};

static int profiler_init = 0;

#if PROFILER_MODE
static int MAX_PROFILER_EVENTS = 0;
static int profiler_count = 0;
static int * profiler_data = 0;
static __int64 * profiler_clock = 0;
static inline __int64 my_rdtsc(){ return __rdtsc(); }
bool ls_getEnvironmentVariable(const char * name, char * value, size_t valueSize)
{
#if _WIN32
	DWORD len = GetEnvironmentVariableA(name, value, (DWORD)valueSize);
	value[valueSize - 1] = 0;
	return (len > 0) ? true : false;
#else
	const char * v = getenv(name);
	if (v) {
		strncpy(value, v, valueSize);
		value[valueSize - 1] = 0;
	}
	return v ? true : false;
#endif
}
const char * header =
"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">\n"
"<html>\n"
"<head>\n"
"<title>%s</title>\n"
"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-1\">\n"
"<meta http-equiv=\"Content-Style-Type\" content=\"text/css\">\n"
"<meta http-equiv=\"Content-Script-Type\" content=\"text/javascript\">\n"
"<style type=\"text/css\">\n"
"  <!--\n"
"  .name0 { font-family: 'Trebuchet MS'; font-size: 15px; text-align: left;  left: 10px; height: 48px; position: fixed; }\n"
"  .name1 { font-family: 'Comic Sans MS'; font-size: 12px; text-align: left;  left: 12px; height: 48px; position: fixed; }\n"
"  .time0 { border: 1px solid #000000; background-color: #2080c0; height: 40px; position: absolute; }\n"
"  .time1 { border: 0px solid #000000; background-color: #ffffff; height: 40px; position: absolute; }\n"
"  -->\n"
"</style>\n"
"<script type=\"text/JavaScript\">\n"
"  function load() {\n"
;

const char * footer =
"  }\n"
"</script>\n"
"</head>\n"
"<table>\n"
"<tr>\n"
"<td><img src = \"https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/AMD_Logo.svg/2000px-AMD_Logo.svg.png\" \n"
"alt = \"RadeonLoom\" style=\"float:center; width:160px; height:60px; \"></td>\n"
"</tr>\n"
"<tr>\n"
"<th><b><strong>inference_server_app</strong> Visual Profiling</b></th>\n"
"</tr>\n"
"</table>\n"
"<body onload='load()'>\n"
"<div style=\"width:1000px;\">\n"
"<div id=\"Canvas\" style=\"position: absolute; left:%dpx; width:%dpx; height:%dpx; border:solid black; overflow:auto;\"></div>\n"
"</div>\n"
"</body>\n"
"</html>\n"
;

const char * colorlist[] = {
	"#66FFFF", "#66FFCC", "#66FF99", "#66FF66", "#66FF33", "#66FF00", "#66CCFF", "#66CCCC",
	"#66CC99", "#66CC66", "#66CC33", "#66CC00", "#6699FF", "#6699CC", "#669999", "#669966",
	"#669933", "#669900", "#6666FF", "#6666CC", "#666699", "#666666", "#666633", "#666600",
	"#6633FF", "#6633CC", "#663399", "#663366", "#663333", "#663300", "#6600FF", "#6600CC",
	"#660099", "#660066", "#660033", "#660000"
};

char default_profiler_name[1024] = "profiler";

static void dump_profile_log()
{
#if _WIN32
    CreateDirectory("AnnInfereceServer-Visual-Profile", NULL);
#else
	struct stat st = {0};
    if (stat("inference_server_app-Visual-Profile", &st) == -1) {	mkdir("inference_server_app-Visual-Profile", 0700); }
#endif
    char profiler[1024] = "inference_server_app-Visual-Profile/AnnInfereceServer-profile";
	char textBuffer[1024];
    if (ls_getEnvironmentVariable("VISUAL_PROFILER_LOCATION", textBuffer, sizeof(textBuffer))){sprintf(profiler, "%s/inference_server_app-profile", textBuffer);}
	char plogfile[1024]; sprintf(plogfile, "%s-data.log", profiler);
	char phtmfile[1024]; sprintf(phtmfile, "%s-visual.html", profiler);
    FILE * fp = fopen(plogfile, "w"); if (!fp) { error("ERROR: unable to create '%s'\n", plogfile); return; }
    FILE * fh = fopen(phtmfile, "w"); if (!fh) { error("ERROR: unable to create '%s'\n", phtmfile); return; }

	fprintf(fh, header, phtmfile);
	int width = 1000;
	int height = 400;
	int nidlist = 0;
	int idlist[PROFILER_NUM_EVENTS][2];
	float tsum[PROFILER_NUM_EVENTS];
	float tmin[PROFILER_NUM_EVENTS];
	float tmax[PROFILER_NUM_EVENTS];
	int tcount[PROFILER_NUM_EVENTS];
	int ncolors = (int)(sizeof(colorlist) / sizeof(colorlist[0]));
	memset(idlist, -1, sizeof(idlist));
	memset(tcount, 0, sizeof(tcount));
	float freq = 0.0;
#if _WIN32
	LARGE_INTEGER li, l2;
	QueryPerformanceFrequency(&li);
	freq = (float)li.QuadPart;
	QueryPerformanceCounter(&li);
	__int64 stime = my_rdtsc();
	Sleep(1000);
	QueryPerformanceCounter(&l2);
	__int64 etime = my_rdtsc();
	freq *= ((float)(etime - stime)) / ((float)(l2.QuadPart - li.QuadPart));
#else
    freq = std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
#endif
	int xstart = 280;
	for (int k = 0; k < profiler_count; k++) {
		int e = profiler_data[k] >> 2;
		int t = profiler_data[k] & 3;
		if (t == 0) {
			for (int j = k + 1; j < profiler_count; j++) {
				if ((profiler_data[k] ^ profiler_data[j]) == 1) {
					if (idlist[e][0] == -1) {
						idlist[e][0] = -2;
					}
				}
			}
		}
	}
	for (int e = 0; e < PROFILER_NUM_EVENTS; e++) {
		if (idlist[e][0] == -2) {
			idlist[e][0] = xstart;
			idlist[e][1] = 100 + nidlist * 50;
			nidlist++;
		}
	}
	int max_time = 10 + (int)((float)(profiler_clock[profiler_count - 1] - profiler_clock[0])*1000.0f / freq);
	for (int k = 0; k <= max_time; k += 20) {
		int barx = xstart + (int)(k * 10);
		fprintf(fh, "    d = document.createElement('div'); d.title = '%d ms'; d.className='time0'; d.style.backgroundColor='#FFFFFF'; d.style.top='%dpx'; d.style.left='%dpx'; d.style.width='%dpx'; document.getElementsByTagName('body')[0].appendChild(d);\n",
			k, 40, barx - 1, 1);
		fprintf(fh, "    d = document.createElement('div'); e = document.createTextNode('%3dms'); d.appendChild(e); d.className='time1'; d.style.backgroundColor='#FFFFFF'; d.style.top='%dpx'; d.style.left='%dpx'; document.getElementsByTagName('body')[0].appendChild(d);\n",
			k, 50, barx + 2);
	}
	for (int k = 0; k < profiler_count; k++) {
		float fclock = (float)(profiler_clock[k] - profiler_clock[0])*1000.0f / freq;
		int e = profiler_data[k] >> 2;
		int t = profiler_data[k] & 3;
		if (t == 0) {
			char data[128]; data[0] = 0;
			for (int j = k + 1; j < profiler_count; j++) {
				if ((profiler_data[k] ^ profiler_data[j]) == 2) {
					union { __int64 a; int b[2]; } u; u.a = profiler_clock[j];
					sprintf(data, " DATA[%d,%d]", u.b[0], u.b[1]);
				}
				if ((profiler_data[k] ^ profiler_data[j]) == 1) {
					float fclockj = (float)(profiler_clock[j] - profiler_clock[0])*1000.0f / freq;
					float start = fclock, duration = fclockj - fclock; int id = e; const char * name = ProfilerEventName[e];
					int barx = xstart + (int)(start * 10);
					int barw = (int)(duration * 10);
					const char * color = colorlist[id%ncolors];
					if ((idlist[id][0] + 6) < barx) {
						fprintf(fh, "    d = document.createElement('div'); d.title = '%s'; d.className='time1'; d.style.top='%dpx'; d.style.left='%dpx'; d.style.width='%dpx'; document.getElementsByTagName('body')[0].appendChild(d);\n",
							name, idlist[id][1], idlist[id][0] + 3, barx - idlist[id][0] - 6);
					}
					fprintf(fh, "    d = document.createElement('div'); d.title = '%s %5.3fms @%5.3fms%s'; d.className='time0'; d.style.backgroundColor='%s'; d.style.top='%dpx'; d.style.left='%dpx'; d.style.width='%dpx'; document.getElementsByTagName('body')[0].appendChild(d);\n",
						name, duration, start, data, color, idlist[id][1], barx, barw);
					if (idlist[id][0] < (barx + barw)) {
						idlist[id][0] = (barx + barw);
					}
					if (width < (barx + barw)) {
						width = barx + barw;
					}
					if (tcount[id] == 0) { tmin[id] = duration; tmax[id] = duration; tsum[id] = 0.0f; }
					tsum[id] += duration;
					if (duration < tmin[id]) tmin[id] = duration;
					if (duration > tmax[id]) tmax[id] = duration;
					tcount[id]++;
					break;
				}
			}
		}
		if (t == 2) {
            fprintf(fp, "DATA %20lld %3d %s\n", (long long int)profiler_clock[k], e, ProfilerEventName[e]);
		}
		else {
			fprintf(fp, "%12.3f %-12s %3d %s\n", fclock, t ? "stop" : "start", e, ProfilerEventName[e]);
		}
	}
	for (int e = 0; e < PROFILER_NUM_EVENTS; e++) {
		if (idlist[e][0] >= 0) {
			fprintf(fh, "    d = document.createElement('div'); e = document.createTextNode('%s'); d.appendChild(e); d.className='name0'; d.style.top='%dpx'; document.getElementsByTagName('body')[0].appendChild(d);\n",
				ProfilerEventName[e], idlist[e][1] + 3);
			fprintf(fh, "    d = document.createElement('div'); e = document.createTextNode('%5.1f(min) %5.1f(avg) %5.1f(max) %3d(count)'); d.appendChild(e); d.className='name1'; d.style.top='%dpx'; document.getElementsByTagName('body')[0].appendChild(d);\n",
				tmin[e], tsum[e] / tcount[e], tmax[e], tcount[e], idlist[e][1] + 3 + 18);
		}
	}
	fclose(fp);
	height = 60 + nidlist * 50 + 100;
	fprintf(fh, footer, xstart - 30, width - xstart + 50, height);
	fclose(fh);
    info("inference_server_app Visual Profile:Dumped profiler log from %d events into %s and %s\n", profiler_count, plogfile, phtmfile);
}

void __stdcall _PROFILER_START(ProfilerEventEnum e)
{
	if (profiler_data && profiler_count < MAX_PROFILER_EVENTS) {
		int k = profiler_count++;
		profiler_data[k] = (((int)e) << 2) | 0;
		profiler_clock[k] = my_rdtsc();
		if (profiler_count == MAX_PROFILER_EVENTS) PROFILER_SHUTDOWN();
	}
}

void __stdcall _PROFILER_STOP(ProfilerEventEnum e)
{
	if (profiler_data && profiler_count < MAX_PROFILER_EVENTS) {
		int k = profiler_count++;
		profiler_data[k] = (((int)e) << 2) | 1;
		profiler_clock[k] = my_rdtsc();
		if (profiler_count == MAX_PROFILER_EVENTS) PROFILER_SHUTDOWN();
	}
}

void __stdcall _PROFILER_DATA(ProfilerEventEnum e, __int64 value)
{
	if (profiler_data && profiler_count < MAX_PROFILER_EVENTS) {
		int k = profiler_count++;
		profiler_data[k] = (((int)e) << 2) | 2;
		profiler_clock[k] = value;
		if (profiler_count == MAX_PROFILER_EVENTS) PROFILER_SHUTDOWN();
	}
}
#endif

void __stdcall PROFILER_INITIALIZE()
{
    info("inference_server_app Visual Profile:Start\n");
	if (profiler_init++ == 0)
	{
		int EnableProfiler = 16000;
		profiler_count = 0;
		if (EnableProfiler) {
			MAX_PROFILER_EVENTS = EnableProfiler;
			if (MAX_PROFILER_EVENTS < 1024) MAX_PROFILER_EVENTS = 1024;
			profiler_data = new int[MAX_PROFILER_EVENTS + 64];
			profiler_clock = new __int64[MAX_PROFILER_EVENTS + 64];
		}
	}
}

void __stdcall PROFILER_SHUTDOWN()
{
    info("inference_server_app Visual Profile:Stop \n");
	if (--profiler_init == 0)
	{
		if (profiler_data)
		{
			dump_profile_log();
			delete profiler_data;
			delete profiler_clock;
			profiler_data = 0;
			profiler_clock = 0;
		}
	}
}

#endif
