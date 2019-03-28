// flucpu256.cpp : Defines the entry point for the application.
//
#include "resource.h"

#include <windows.h>
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>
#include <gl/glew.h>
#include <math.h>
#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include "vec.h"
#include "qustate.h"
#include <opencv2/opencv.hpp>

#include <vector>
#include <algorithm>

#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"cusolver.lib")
#pragma comment(lib,"cufft.lib")
#pragma comment(lib,"opencv_world401.lib")


#define MAX_LOADSTRING 100
#define PI 3.14159265358979324

const int size = 4096;
const int spat = 32768;
 
struct eigent {
	current value;
	current2 *vec;
};

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING] = L"qustate";                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING] = L"w32";            // the main window class name

HDC hdc1, hdc2;
HGLRC m_hrc;
int mx, my, cx, cy;
double ang1, ang2, len, cenx, ceny, cenz;
GLuint v1, imgtex;
GLuint vbo[2], pbo;
int viewm = 0;
GLuint shader, program;

const double l = -8, r = 8;
float v[spat];
double2 potex[spat];
double2 potek[spat];
double2 ws[spat];
float3 pline[(spat - 1) * 2];
current2 *hm;
current2 *eigenvec;
current eigenval[size];
//float3 lines[(size - 1) * 2 * size];
int ct[4096];
//float ldos[size*size];
//int img[size*size];


double el = size / (r - l) / 23;
double xl = size / (r - l) / 23;
double br = size * 4096 / (r - l) / (r - l) / 1536;
double es = 0.005;
double eb = 0;
int ise = 0, iseb = 0, isx = 0;
int viewc = 512;
float height = 1.0;
int ish = 0;
int upline=0, upimg=0;

float mat[16] = {
	1,0,0,0,
	0,0,-height*0.125,0,
	0,0,1,0,
	0,0,0,1,
};

int fileindex(void) {
	FILE *fi;
	int index = 0;
	if (!fopen_s(&fi, "D:/files/courses/qu/img/index.txt", "rb")) {
		fscanf(fi, "%d", &index);
		fclose(fi);
	}
	if (!fopen_s(&fi, "D:/files/courses/qu/img/index.txt", "wb")) {
		fprintf(fi, "%d", index + 1);
		fclose(fi);
	}
	return index;
}

cv::Mat frame(size, size, CV_8UC3, cv::Scalar(0, 0, 0));

char sf[1024] =
"float a;\r\n"
"float l = 0.25;\r\n"
"a = x / l - floor(x / l);\r\n"
"float i = ((7.0 / 16.0 < a) && (a < 9.0 / 16.0))?1.0:0.0;\r\n"
"return i*(x*x<1.0?1.0:2.0) * 1000.0;"
;

const char *comsha[3] = {
"#version 430 core\n"
"layout(local_size_x = 32) in;"
"layout(r32f, binding = 0) uniform image1D v1;"
"uniform float l,dx;"
"float f(float x){"
,sf,
"}"
"void main() {"
	"float x = l+dx*float(gl_GlobalInvocationID.x);"
	"float a=f(x);"
	"imageStore(v1,int(gl_GlobalInvocationID.x), vec4(a));"
"}"
};


unsigned char getr(double x) {
	return (tanh((x - 0.375) * 6) + 1) * 127;
}
unsigned char getg(double x) {
	return (tanh((x - 0.625) * 6) + 1) * 127;
}
unsigned char getb(double x) {
	return (exp(-20 * (x - 0.25)*(x - 0.25) - 2.0*exp(-(x + 0.05)*(x + 0.05)*144.0)) *0.5 + 1 + tanh((x - 0.875) * 6)) * 255 / 2;
}
bool cp(eigent a, eigent b) {
	return a.value < b.value;
}

int reorder(int size, int i) {
	return size / 2 + ((i % 2) ? 1 : -1)*(size - i) / 2;
}

int cui(int size, GLuint vbo, GLuint pbo,int *ct);
int cueigen(void *hm, void *val, void *vec, int size);
int cuimg(float *img, int size, float m, float el, float xl, float es, float eb, float br);
int culine(int size, float l, float r, float es, float eb, float height);

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK editkernal(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam);
INT_PTR CALLBACK videokernal(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam);

int fft(double2 *in, double2 *out, double2 *ws, int n) {
	int i, j, k, l;
	double2 a, b;
	j = 0;
	for (i = 0; i < n - 1; i++)
	{
		ws[i].x = cos((double)i / n * PI);
		ws[i].y = sin((double)i / n * PI);
		out[j] = in[i];
		k = n / 2;
		while (j >= k) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}
	ws[i].x = cos((double)i / n * PI);
	ws[i].y = sin((double)i / n * PI);
	out[j] = in[i];
	i = 1;
	while (i < n) {
		j = n / i;
		for (k = 0; k < n; k += 2 * i) {
			for (l = 0; l < i; l++) {
				a = out[k + l] + out[k + l + i] * ws[l*j];
				b = out[k + l] - out[k + l + i] * ws[l*j];
				out[k + l] = a;
				out[k + l + i] = b;
			}
		}
		i *= 2;
	}
	return 0;
}

int gimg(void) {
	int i, j;
	if (xl > 64) {
		xl = 64;
	}
	cuimg(NULL, size, 1.0 / (r - l), el, xl, es, eb, br);
	/*for (i = 0; i < size*size; i++) {
		j = log(ldos[i] * br / el + 1.0) * 2048 + 1024;
		if (j > 4095) {
			j = 4095;
		}
		if (j < 0) {
			j = 0;
		}
		img[i] = ct[j];
	}*/
	glBindTexture(GL_TEXTURE_2D, imgtex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	return 0;
}

int gline(void) {
	int i, j;
	float dx = (r - l)*(1.0 / spat);
	for (j = 0; j < spat - 1; j++) {
		pline[j * 2 + 0] = { l + dx * (j + 0.5),potex[j].x * 0.001,0.0 };
		pline[j * 2 + 1] = { l + dx * (j + 1.5),potex[j + 1].x * 0.001,0.0 };
	}
	/*dx = (r - l)*(1.0 / size);
	for (i = 0; i < size; i++) {
		for (j = 0; j < size - 1; j++) {
			float y = (eigen[i].value - eb)*es;
			lines[i*(size - 1) * 2 + j * 2 + 0] = { l + dx * (j + 0.5f),norm(eigen[i].vec[j])*height / (r - l),-y };
			lines[i*(size - 1) * 2 + j * 2 + 1] = { l + dx * (j + 1.5f),norm(eigen[i].vec[j + 1])*height / (r - l),-y };
			//lines[i*(size - 1) * 2 + j * 2 + 0] = { l + dx * (j + 0.5),eigen[i].vec[j].x / sqrt(dx*size)*0.1,eigen[i].value.x*0.001+ eigen[i].vec[j].y / sqrt(dx*size)*0.1 };
			//lines[i*(size - 1) * 2 + j * 2 + 1] = { l + dx * (j + 1.5),eigen[i].vec[j + 1].x / sqrt(dx*size)*0.1,eigen[i].value.x*0.001+ eigen[i].vec[j+1].y / sqrt(dx*size)*0.1 };
		}
	}*/
	//glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	//glBufferData(GL_ARRAY_BUFFER, (size - 1) * 2 * size * sizeof(float3), lines, GL_STATIC_DRAW);
	culine(size, l, r, es, eb, height);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, (spat - 1) * 2 * sizeof(float3), pline, GL_STATIC_DRAW);
	return 0;
}

int caleigen(float k0) {
	int i, j;
	for (j = 0; j < spat; j++) {
		potex[j] = { v[j],0 };
	}

	fft(potex, potek, ws, spat);
	for (j = 0; j < size; j++) {
		int j1 = (j >= size / 2) ? spat - size + j : j;
		for (i = 0; i < size; i++) {
			int i1 = (i >= size / 2) ? spat - size + i : i;
			current k;
			k = (i > size / 2) ? i - size : i;
			k = k0 + k * 2 * PI / (r - l);
			hm[j*size + i] = { potek[(j1 - i1 + spat) % spat].x*(1.0 / spat) + ((i == j) ? k * k *0.5 : 0),potek[(j1 - i1 + spat) % spat].y*(1.0 / spat) };
		}
	}
	cueigen(hm, eigenval, eigenvec, size);

	return 0;
}

int calc(float k0) {
	int i, j;

	glUseProgram(program);
	glUniform1f(glGetUniformLocation(program, "l"), l);
	glUniform1f(glGetUniformLocation(program, "dx"), (r - l)*(1.0 / spat));
	glBindImageTexture(0, v1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

	glDispatchCompute((spat + 31) / 32, 1, 1);
	glFinish();
	glBindTexture(GL_TEXTURE_1D, v1);
	glGetTexImage(GL_TEXTURE_1D, 0, GL_RED, GL_FLOAT, v);
	caleigen(k0);
	//std::sort(eigen.begin(), eigen.end(), cp);
	gline();
	gimg();
	return 0;
}

void draw(void) {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(0x00004100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(len * cos(ang1)*cos(ang2) + cenx, len * sin(ang2) + ceny, len * sin(ang1)*cos(ang2) + cenz, cenx, ceny, cenz, 0, cos(ang2), 0);


	glBindTexture(GL_TEXTURE_2D, 0);
	glBegin(GL_LINES);
	glColor3f(1.0, 1.0, 0);
	glVertex3f(1.0f, 0.0f, -0.0f);
	glVertex3f(-1.0f, 0.0f, -0.0f);
	glVertex3f(0.0f, 1.0f, -0.0f);
	glVertex3f(0.0f, -1.0f, -0.0f);
	glVertex3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, -1.0f);
	glEnd();
	
	if (viewm == 2|| viewm == 3) {
		glColor3f(1.0, 1.0, 1.0);
	}
	else {
		glColor3f(0.0, 0.0, 1.0);
	}
	if (viewm == 0 || viewm == 2|| viewm == 3) {
		glPushMatrix();
		if (viewm == 3) {
			mat[6] = -0.125*height;
			glMultMatrixf(mat);
		}
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glVertexPointer(3, GL_FLOAT, 0, NULL);
		glDrawArrays(GL_LINES, 0, (size - 1) * 2 * ((size > viewc) ? viewc : size));
		glPopMatrix();
	}
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glVertexPointer(3, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_LINES, 0, (spat - 1) * 2);

	if (viewm == 0 || viewm == 1) {
		glBindTexture(GL_TEXTURE_2D, imgtex);
		glBegin(GL_QUADS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2f(0.0f, 0.0f);
		glVertex3f(l, 0, 0);
		glTexCoord2f(1.0f, 0.0f);
		glVertex3f(r, 0, 0);
		glTexCoord2f(1.0f, 1.0f);
		glVertex3f(r, 0, l - r);
		glTexCoord2f(0.0f, 1.0f);
		glVertex3f(l, 0, l - r);
		glEnd();
	}
	SwapBuffers(wglGetCurrentDC());
}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow)
{
	int i, j;
	char s[256];
	unsigned int t, t1, count, f;
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	// TODO: Place code here.
	MyRegisterClass(hInstance);

	// Perform application initialization:
	if (!InitInstance(hInstance, nCmdShow))
	{
		return FALSE;
	}

	MSG msg;
	t = GetTickCount();
	count = 0;
	f = 0;
	// Main message loop:	
st:
	for (;;) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)){
			if (msg.message == WM_QUIT)
				break;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else {
			if (upimg) {
				gimg();
				upimg = 0;
			}
			if (upline) {
				gline();
				upline = 0;
			}
			draw();
		}
	}
	return (int)msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEXW wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, NULL);
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDR_MENU1);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, NULL);

	return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance; // Store instance handle in our global variable

	HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return TRUE;
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	break;
	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps);
		// TODO: Add any drawing code that uses hdc here...
		draw();
		EndPaint(hWnd, &ps);
	}
	break;
	case WM_DESTROY: {
		PostQuitMessage(0);
		break;
	}
	case WM_CREATE: {
		int i, j;
		char s[1024];
		PIXELFORMATDESCRIPTOR pfd = {
			sizeof(PIXELFORMATDESCRIPTOR),
			1,
			PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_STEREO,
			PFD_TYPE_RGBA,
			24,
			0,0,0,0,0,0,0,0,
			0,
			0,0,0,0,
			32,
			0,0,
			PFD_MAIN_PLANE,
			0,0,0,0
		};
		hdc1 = GetDC(hWnd);
		hdc2 = GetDC(NULL);
		int uds = ::ChoosePixelFormat(hdc1, &pfd);
		::SetPixelFormat(hdc1, uds, &pfd);
		m_hrc = ::wglCreateContext(hdc1);
		::wglMakeCurrent(hdc1, m_hrc);
		glewInit();
		glDisable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnable(GL_TEXTURE_1D);
		glEnable(GL_TEXTURE_2D);
		//glEnable(GL_ALPHA_TEST);
		//glAlphaFunc(GL_GREATER, 0.0f);
		//glEnableClientState(GL_NORMAL_ARRAY);
		//((bool(_stdcall*)(int))wglGetProcAddress("wglSwapIntervalEXT"))(1);

		ang1 = PI * 0.5;
		ang2 = 0.7;
		len = 20;
		cenx = 0.0;
		ceny = 0.0;
		cenz = 0.0;

		hm = (current2*)malloc(size*size * sizeof(current2));
		eigenvec = (current2*)malloc(size*size * sizeof(current2));

		glGenBuffers(2, vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, (size - 1) * 2 * size * sizeof(float3), NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, (spat - 1) * 2 * sizeof(float3), NULL, GL_STATIC_DRAW);

		glGenBuffers(1, &pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size*size * 4, NULL, GL_STREAM_DRAW_ARB);

		glGenTextures(1, &v1);
		glBindTexture(GL_TEXTURE_1D, v1);
		glTexStorage1D(GL_TEXTURE_1D, 1, GL_R32F, spat);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP);

		glGenTextures(1, &imgtex);
		glBindTexture(GL_TEXTURE_2D, imgtex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
		
		GLfloat fLargest;
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest); 
//		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);

		for (i = 0; i < 4096; i++) {
			ct[i] = (i > (1024 + 2) ? (255 << 24) : 0) + (getb((double)i / 2048 - 0.5) << 16) + (getg((double)i / 2048 - 0.5) << 8) + getr((double)i / 2048 - 0.5);
		}

		cui(size, vbo[0], pbo, ct);

		shader = glCreateShader(GL_COMPUTE_SHADER);
		glShaderSource(shader, 3, comsha, NULL);
		glCompileShader(shader);
		glGetInfoLogARB(shader, 1024, &i, s);
		if (i) {
			MessageBox(hWnd, s, "error", MB_OK);
		}

		program = glCreateProgram();
		glAttachShader(program, shader);
		glLinkProgram(program);
		glGetInfoLogARB(program, 1024, &i, s);
		if (i) {
			MessageBox(hWnd, s, "error", MB_OK);
		}

		calc(0);
		break;
	}
	case WM_SIZE: {
		cx = lParam & 0xffff;
		cy = (lParam & 0xffff0000) >> 16;
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-(float)cx / (cx + cy) *len*0.0078125, (float)cx / (cx + cy) *len*0.0078125, -(float)cy / (cx + cy)*len*0.0078125, (float)cy / (cx + cy) *len*0.0078125, len*0.00390625 * 4, len*400.0);
		glViewport(0, 0, cx, cy);
		break;
	}
	case WM_MOUSEMOVE: {
		int x, y, f;
		f = 0;
		x = (lParam & 0xffff);
		y = ((lParam & 0xffff0000) >> 16);
		if (MK_LBUTTON&wParam) {
			f = 1;
			ang1 += (x - mx)*0.002;
			ang2 += (y - my)*0.002;
		}
		if (MK_RBUTTON&wParam) {
			double l;
			f = 1;
			l = len * 1.25 / (cx + cy);
			cenx += l * (-(x - mx)*sin(ang1) - (y - my)*sin(ang2)*cos(ang1));
			ceny += l * ((y - my)* cos(ang2));
			cenz += l * ((x - mx)*cos(ang1) - (y - my)*sin(ang2)*sin(ang1));
		}
		mx = x;
		my = y;
		if (f) {
			//draw();
		}
		break;
	}
	case WM_MOUSEWHEEL: {
		short m;
		m = (wParam & 0xffff0000) >> 16;
		if (ish) {
			height *= exp(m * 0.001);
			upline = 1;
		}
		if (ise) {
			es *= exp(m * 0.001);
			el *= exp(m * 0.001);
			br *= exp(m * 0.001);
			if (el < 1.0) {
				el = 1.0;
			}
			upline = 1;
			upimg = 1;
		}
		if (isx) {
			xl *= exp(m * 0.001);
			if (xl > 64) {
				xl = 64;
			}
			if (xl < 1.0) {
				xl = 1.0;
			}
			upimg = 1;
		}
		if (iseb) {
			eb += m / 120.0 / es;
			upline = 1;
			upimg = 1;
		}
		if (!(ish || ise || iseb || isx)) {
			len *= exp(-m * 0.001);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glFrustum(-(float)cx / (cx + cy) *len*0.0078125, (float)cx / (cx + cy) *len*0.0078125, -(float)cy / (cx + cy)*len*0.0078125, (float)cy / (cx + cy) *len*0.0078125, len*0.00390625 * 4, len*400.0);
			//draw();
			break;

		}
	}
	case WM_KEYDOWN: {
		switch (wParam) {
		case 'N': {
			DialogBox(hInst, MAKEINTRESOURCE(IDD_DIALOG1), hWnd, editkernal);
			break;
		}
		case 'V': {
			viewm = (viewm + 1) % 4;
			//draw();
			break;
		}
		case 'Q': {
			el *= sqrt(2.0);
			upimg = 1;
			break;
		}
		case 'A': {
			el *= sqrt(0.5);
			if (el < 1.0) {
				el = 1.0;
			}
			upimg = 1;
			break;
		}
		case 'E': {
			br *= sqrt(2.0);
			upimg = 1;
			break;
		}
		case 'D': {
			br *= sqrt(0.5);
			upimg = 1;
			break;
		}
		case 'R': {
			viewc *= 2;
			//draw();
			break;
		}
		case 'F': {
			if (viewc > 128) {
				viewc /= 2;
			}
			//draw();
			break;
		}
		case 'S': {
			int i, j;
			glBindTexture(GL_TEXTURE_2D, imgtex);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
			flip(frame, frame, 0);
			int index = fileindex();
			char info[1024];
			sprintf_s(info, "D:/files/courses/qu/img/%d.png", index);
			cv::imwrite(info, frame);
			sprintf_s(info, "D:/files/courses/qu/img/%d.csv", index);
			FILE *fi;
			if (!fopen_s(&fi, info, "wb")) {
				fprintf(fi, "Ek,");
				for (i = 0; i < size; i++) {
					fprintf(fi, "%f,", l+(i+0.5)*(r-l)/size);
				}
				fprintf(fi, "\n");
				for (i = 0; i < size; i++) {
					fprintf(fi, "%f,", eigenval[i]);
					for (j = 0; j < size; j++) {
						fprintf(fi, "%f,", norm(eigenvec[i*size + j]));
					}
					fprintf(fi, "\n");
				}
				fclose(fi);
			}
			break;
		}
		case 'H': {
			ish = 1;
			break;
		}
		case 'Z': {
			ise = 1;
			break;
		}
		case 'B': {
			iseb = 1;
			break;
		}
		case 'I': {
			el = size / (r - l) / 23;
			xl = size / (r - l) / 23;
			br = size * 4096 / (r - l) / (r - l) / 1536;
			int viewc = 512;
			es = 0.005;
			eb = 0;
			height = 1.0;
			upline = 1;
			upimg = 1;
			break;
		}
		case 'T': {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0);
			break;
		}
		case 'P': {
			DialogBox(hInst, MAKEINTRESOURCE(IDD_DIALOG3), hWnd, videokernal);
			break;
		}
		case 'X': {
			isx = 1;
			break;
		}
		case 'K': {
			int i, j;
			current *Ek;
			int ss = 128;
			Ek = (current*)malloc(ss*ss * sizeof(current));
			for (i = 0; i < ss; i++) {
				caleigen(PI * 2.0 / (r - l) * (i - ss / 2) / ss);
				memcpy(Ek + ss * i, eigenval, ss * sizeof(current));
			}
			FILE *fi;
			int index = fileindex();
			char info[1024];
			sprintf_s(info, "D:/files/courses/qu/img/%d.csv", index);
			if (!fopen_s(&fi, info, "wb")) {
				/*fprintf(fi, "Ek\t");
				for (i = 0; i < ss; i++) {
					fprintf(fi, "n:%d\t", i + 1);
				}
				fprintf(fi, "\n");*/
				for (i = 0; i < ss; i++) {
					fprintf(fi, "%f,", PI * 2.0 / (r - l) * (i - ss / 2) / ss);
					for (j = 0; j < ss; j++) {
						fprintf(fi, "%f,", Ek[i*ss + j]);
					}
					fprintf(fi, "\n");
				}
				fclose(fi);
			}
			free(Ek);
			break;
		}
		}
		break;
	}
	case WM_KEYUP: {
		switch (wParam) {
		case 'H': {
			ish = 0;
			break;
		}
		case 'Z': {
			ise = 0;
			break;
		}
		case 'B': {
			iseb = 0;
			break;
		}
		case 'T': {
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0);
			break;
		}
		case 'X': {
			isx = 0;
			break;
		}
		}
		break;
	}
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}

INT_PTR CALLBACK videokernal(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG: {
		SetDlgItemTextA(hDlg, IDC_EDIT1, sf);
		return (INT_PTR)TRUE;
	}
	case WM_COMMAND: {
		switch (LOWORD(wParam)) {
		case IDOK:
		{
			int i, j;
			char s[1024],info[1024];
			char ls[128], rs[128], ns[128];
			float tl, tr;
			int n;
			GetDlgItemTextA(hDlg, IDC_EDIT1, s, 1024);
			GetDlgItemTextA(hDlg, IDC_EDIT3, ls, 128);
			GetDlgItemTextA(hDlg, IDC_EDIT4, rs, 128);
			GetDlgItemTextA(hDlg, IDC_EDIT5, ns, 128);

			i = sscanf_s(ls, "%f", &tl);
			if (!i || i == 0xffffffff) {
				goto next;
			}
			i = sscanf_s(rs, "%f", &tr);
			if (!i || i == 0xffffffff) {
				goto next;
			}
			i = sscanf_s(ns, "%d", &n);
			if (!i || i == 0xffffffff) {
				goto next;
			}

			int index = fileindex();
			sprintf_s(info, "D:/files/courses/qu/img/%d.mp4", index);
			
			//cv::VideoWriter writer(info, cv::VideoWriter::fourcc('H', '2', '6', '4'), 60.0, cv::Size(size, size));
			cv::VideoWriter writer(info, cv::VideoWriter::fourcc('H', '2', '6', '4'), 60.0, cv::Size(1024, 1024)); 
			cv::Mat frame1(1024, 1024, CV_8UC3, cv::Scalar(0, 0, 0));
			for (int i = 0; i <= n; i++) {
				sprintf_s(sf, "float t=%f;\r\n%s", i*((tr - tl) / n) + tl, s);
				glShaderSource(shader, 3, comsha, NULL);
				glCompileShader(shader);
				glGetInfoLogARB(shader, 1024, &j, info);
				if (j) {
					MessageBox(hDlg, info, "error", MB_OK);
					writer.release();
					goto next;
				}

				program = glCreateProgram();
				glAttachShader(program, shader);
				glLinkProgram(program);
				glGetInfoLogARB(program, 1024, &j, info);
				if (j) {
					MessageBox(hDlg, info, "error", MB_OK);
					writer.release();
					goto next;
				}
				calc(0);
				draw();
				glBindTexture(GL_TEXTURE_2D, imgtex);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
				for (int j = 0; j < 1024; j++) {
					memcpy(frame1.data + 1024 * 3 * j, frame.data + size * 3 * (1023 - j) + 3 * (size - 1024) / 2, 1024 * 3);
				}
				writer << frame1;
			}
			writer.release();
		}
		case IDCANCEL:
		{
		next:
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		}
		break;
	}
	}
	return (INT_PTR)FALSE;
}

INT_PTR CALLBACK editkernal(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG: {
		SetDlgItemTextA(hDlg, IDC_EDIT1, sf);
		return (INT_PTR)TRUE;
	}
	case WM_COMMAND: {
		switch (LOWORD(wParam)) {
		case IDOK:
		{
			int i;
			char s[1024];
			GetDlgItemTextA(hDlg, IDC_EDIT1, sf, 1024);
			glShaderSource(shader, 3, comsha, NULL);
			glCompileShader(shader);
			glGetInfoLogARB(shader, 1024, &i, s);
			if (i) {
				MessageBox(hDlg, s, "error", MB_OK);
				goto next;
			}

			program = glCreateProgram();
			glAttachShader(program, shader);
			glLinkProgram(program);
			glGetInfoLogARB(program, 1024, &i, s);
			if (i) {
				MessageBox(hDlg, s, "error", MB_OK);
				goto next;
			}
			calc(0);
			draw();
		}
		case IDCANCEL:
		{
		next:
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		}
		break;
	}
	}
	return (INT_PTR)FALSE;
}
