// quevo.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "quevo.h"
#include <gl/glew.h>
#include <stdio.h>
#include <math.h>
#include "vec.h"
#include <opencv2/opencv.hpp>


#pragma comment(lib,"opencv_world401.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"glew32.lib")

#define MAX_LOADSTRING 100
#define PI 3.14159265358979324

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

HDC hdc1, hdc2;
HGLRC m_hrc;
GLuint versha, frasha, pro;
int mx, my, cx, cy;
double ang1, ang2, len, cenx, ceny, cenz;
int start=0;
GLuint texbuffer, framebuffer, depthbuffer;

const int seq = 8;
const int size = 4096;
const double l = -8, r =8;
const double ri=0.025;
float3 point[(size - 1) * 6 * seq], pointn[(size - 1) * 6 * seq], popoint[size];
double2 v1[size];
double2 v2[size];
double po[size];
double2 ws[size*6];
unsigned char frim[2048 * 2048 * 3];

cv::VideoWriter writer;
cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));

int fileindex(void) {
	FILE *fi;
	int index = 0;
	if (!fopen_s(&fi, "D:/files/courses/qu/img/index.txt", "rb")) {
		fscanf_s(fi, "%d", &index);
		fclose(fi);
	}
	if (!fopen_s(&fi, "D:/files/courses/qu/img/index.txt", "wb")) {
		fprintf(fi, "%d", index + 1);
		fclose(fi);
	}
	return index;
}


const char *shaderv =
"varying vec4 fc;"
"varying vec4 a;"
"float l;"
"varying vec3 ntemp;"
"void main() {"
"	a = gl_Vertex;"
"	ntemp = gl_Normal;"
"	ntemp = gl_NormalMatrix * ntemp;"
"	a = gl_ModelViewMatrix * a;"
"	fc = gl_Color;"
"	gl_Position = gl_ProjectionMatrix * a;"
"}";

const char *shaderf =
"varying vec4 fc;"
"float s,t,u;"
"vec3 texdir;"
"varying vec4 a;"
"vec3 b,c;"
"varying vec3 ntemp;"
"void main(){"
"	b=vec3(a);"
"	s=b.x*b.x+b.y*b.y+b.z*b.z;"
"	s=1.0/sqrt(s);"
"	b.x=b.x*s;"
"	b.y=b.y*s;"
"	b.z=b.z*s;"
"	c=ntemp;"
"	s=c.x*c.x+c.y*c.y+c.z*c.z;"
"	s=1.0/sqrt(s);"
"	c.x=c.x*s;"
"	c.y=c.y*s;"
"	c.z=c.z*s;"
"	texdir=reflect(b,c);"
"	s=texdir.x*0.276-texdir.y*0.276+texdir.z*0.920;"
"	t=c.x*0.276+c.y*0.276+c.z*0.920;"
"	t=(gl_FrontFacing)?t:-t;"
"	t=(t>0.0)?t:0.0;"
"	s=(s>0.0)?s:0.0;"
"	s=s*s;"
"	s=s*s;"
"	s=s*s;"
"	s=s*0.7+t*0.30+0.05;"
"	s=pow(s,0.5);"
"	gl_FragColor.r = fc.r*s;"
"	gl_FragColor.g = fc.g*s;"
"	gl_FragColor.b = fc.b*s;"
"	gl_FragColor.a = fc.a;"
"}";

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

template <class T>
int solve3(T *m, T *y, T *x, int n, T *workspace);

int evo(double2 *v1, double2 *v2, double2 *ws, double dt, double dx, int size);

double3 rotate(double3 f, double3 t, double3 r) {
	double3 n=f+t;
	return r - n * (dot(r, n) * 2.0 / norm(n));
}

int gettri(double2 *line, float3 *tri,float3 *norms,double l,double r,int count,int seq,double ri) {
	int i,j;
	double3 dir0, dir1;
	double3 ex0, ey0, ex1, ey1;
	double d = (r-l)/ count;
	dir0 = { d,line[1].x - line[0].x,line[1].y - line[0].y };
	dir0 = dir0 * (1.0 / sqrt(norm(dir0)));
	ex0 = rotate({ 1.0,0.0,0.0 }, dir0, { 0.0,1.0,0.0 });
	ey0 = rotate({ 1.0,0.0,0.0 }, dir0, { 0.0,0.0,1.0 });
	for (i = 0; i < size - 2; i++) {
		double x0 = l + (i + 0.5)*((r - l) / count);
		double x1 = l + (i + 1.5)*((r - l) / count);
		dir1 = { d * 2, line[i + 2].x - line[i].x, line[i + 2].y - line[i].y };
		dir1 = dir1 * (1.0 / sqrt(norm(dir1)));
		ex1 = rotate({ 1.0,0.0,0.0 }, dir1, { 0.0,1.0,0.0 });
		ey1 = rotate({ 1.0,0.0,0.0 }, dir1, { 0.0,0.0,1.0 });
		for (j = 0; j < seq; j++) {
			norms[i * 6 * seq + j * 6 + 0] = ex0 * cos(j * 2 * PI / seq) + ey0 * sin(j * 2 * PI / seq);
			tri[i * 6 * seq + j * 6 + 0] = double3{ x0,line[i].x,line[i].y }+norms[i * 6 * seq + j * 6 + 0] * ri;
			norms[i * 6 * seq + j * 6 + 1] = ex0 * cos((j + 1) * 2 * PI / seq) + ey0 * sin((j + 1) * 2 * PI / seq);
			tri[i * 6 * seq + j * 6 + 1] = double3{ x0,line[i].x,line[i].y }+norms[i * 6 * seq + j * 6 + 1] * ri;
			norms[i * 6 * seq + j * 6 + 2] = ex1 * cos((j + 1) * 2 * PI / seq) + ey1 * sin((j + 1) * 2 * PI / seq);
			tri[i * 6 * seq + j * 6 + 2] = double3{ x1,line[i + 1].x,line[i + 1].y }+norms[i * 6 * seq + j * 6 + 2] * ri;
			norms[i * 6 * seq + j * 6 + 3] = ex0 * cos(j * 2 * PI / seq) + ey0 * sin(j * 2 * PI / seq);
			tri[i * 6 * seq + j * 6 + 3] = double3{ x0,line[i].x,line[i].y }+norms[i * 6 * seq + j * 6 + 3] * ri;
			norms[i * 6 * seq + j * 6 + 4] = ex1 * cos((j + 1) * 2 * PI / seq) + ey1 * sin((j + 1) * 2 * PI / seq);
			tri[i * 6 * seq + j * 6 + 4] = double3{ x1,line[i + 1].x,line[i + 1].y }+norms[i * 6 * seq + j * 6 + 4] * ri;
			norms[i * 6 * seq + j * 6 + 5] = ex1 * cos(j * 2 * PI / seq) + ey1 * sin(j * 2 * PI / seq);
			tri[i * 6 * seq + j * 6 + 5] = double3{ x1,line[i + 1].x,line[i + 1].y }+norms[i * 6 * seq + j * 6 + 5] * ri;
		}
		dir0 = dir1;
		ex0 = ex1;
		ey0 = ey1;
	}
	double x0 = l + (i + 0.5)*((r - l) / count);
	double x1 = l + (i + 1.5)*((r - l) / count);
	dir1 = { d, line[i + 1].x - line[i].x, line[i + 1].y - line[i].y };
	dir1 = dir1 * (1.0 / sqrt(norm(dir1)));
	ex1 = rotate({ 1.0,0.0,0.0 }, dir1, { 0.0,1.0,0.0 });
	ey1 = rotate({ 1.0,0.0,0.0 }, dir1, { 0.0,0.0,1.0 });
	for (j = 0; j < seq; j++) {
		norms[i * 6 * seq + j * 6 + 0] = ex0 * cos(j * 2 * PI / seq) + ey0 * sin(j * 2 * PI / seq);
		tri[i * 6 * seq + j * 6 + 0] = double3{ x0,line[i].x,line[i].y }+norms[i * 6 * seq + j * 6 + 0] * ri;
		norms[i * 6 * seq + j * 6 + 1] = ex0 * cos((j + 1) * 2 * PI / seq) + ey0 * sin((j + 1) * 2 * PI / seq);
		tri[i * 6 * seq + j * 6 + 1] = double3{ x0,line[i].x,line[i].y }+norms[i * 6 * seq + j * 6 + 1] * ri;
		norms[i * 6 * seq + j * 6 + 2] = ex1 * cos((j + 1) * 2 * PI / seq) + ey1 * sin((j + 1) * 2 * PI / seq);
		tri[i * 6 * seq + j * 6 + 2] = double3{ x1,line[i + 1].x,line[i + 1].y }+norms[i * 6 * seq + j * 6 + 2] * ri;
		norms[i * 6 * seq + j * 6 + 3] = ex0 * cos(j * 2 * PI / seq) + ey0 * sin(j * 2 * PI / seq);
		tri[i * 6 * seq + j * 6 + 3] = double3{ x0,line[i].x,line[i].y }+norms[i * 6 * seq + j * 6 + 3] * ri;
		norms[i * 6 * seq + j * 6 + 4] = ex1 * cos((j + 1) * 2 * PI / seq) + ey1 * sin((j + 1) * 2 * PI / seq);
		tri[i * 6 * seq + j * 6 + 4] = double3{ x1,line[i + 1].x,line[i + 1].y }+norms[i * 6 * seq + j * 6 + 4] * ri;
		norms[i * 6 * seq + j * 6 + 5] = ex1 * cos(j * 2 * PI / seq) + ey1 * sin(j * 2 * PI / seq);
		tri[i * 6 * seq + j * 6 + 5] = double3{ x1,line[i + 1].x,line[i + 1].y }+norms[i * 6 * seq + j * 6 + 5] * ri;
	}
	return 0;
}

void draw(void) {
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glClear(0x00004100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(len * cos(ang1)*cos(ang2) + cenx, len * sin(ang2) + ceny, len * sin(ang1)*cos(ang2) + cenz, cenx, ceny, cenz, 0, cos(ang2), 0);

	glColor3f(0x66 * (1.0 / 255), 0xcc * (1.0 / 255), 0xff * (1.0 / 255));
	glVertexPointer(3, GL_FLOAT, 0, point);
	glNormalPointer(GL_FLOAT, 0, pointn);
	glDrawArrays(GL_TRIANGLES, 0, (size - 1) * 6 * seq);

	glBegin(GL_LINES);
	glColor3f(0, 0, 0);
	glVertex3f(1.0f, 0.0f, -0.0f);
	glVertex3f(-1.0f, 0.0f, -0.0f);
	glVertex3f(0.0f, 1.0f, -0.0f);
	glVertex3f(0.0f, -1.0f, -0.0f);
	glVertex3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, -1.0f);
	glEnd();

	glVertexPointer(3, GL_FLOAT, 0, popoint);
	glDrawArrays(GL_LINE_STRIP, 0, size);
}


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	// TODO: Place code here.

	// Initialize global strings
	LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadStringW(hInstance, IDC_QUEVO, szWindowClass, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	// Perform application initialization:
	if (!InitInstance(hInstance, nCmdShow))
	{
		return FALSE;
	}

	HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_QUEVO));

	MSG msg;

	// Main message loop:

st:
	if (start) {
		for (;;) {
			if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				if (msg.message == WM_QUIT)
					break;
				if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
				{
					TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
			}

			else if (start) {
				int i, j;
				for (i = 0; i < 20; i++) {
					evo(v1, v2, ws, 0.1 / 60 / 20 / 2, (r - l) / size, size);
					evo(v2, v1, ws, 0.1 / 60 / 20 / 2, (r - l) / size, size);
				}
				gettri(v1, point, pointn,l, r, size, seq, ri);

				glViewport(0, 0, 1920, 1080);
				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glFrustum(-(float)1920 / (1920 + 1080) *len*0.0078125, (float)1920 / (1920 + 1080) *len*0.0078125, -(float)1080 / (1920 + 1080)*len*0.0078125, (float)1080 / (1920 + 1080) *len*0.0078125, len*0.00390625, len*100.0);
				draw();
				glFinish();
				
				glViewport(0, 0, cx, cy);
				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glFrustum(-(float)cx / (cx + cy) *len*0.0078125, (float)cx / (cx + cy) *len*0.0078125, -(float)cy / (cx + cy)*len*0.0078125, (float)cy / (cx + cy) *len*0.0078125, len*0.00390625, len*100.0);

				draw();
				SwapBuffers(hdc1);

				glBindTexture(GL_TEXTURE_2D, texbuffer);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, frim);
				for (i = 0; i < 1080; i++) {
					for (j = 0; j < 1920; j++) {
						frame.data[(i * 1920 + j) * 3 + 0] = frim[((1079 - i) * 2048 + j) * 3 + 2];
						frame.data[(i * 1920 + j) * 3 + 1] = frim[((1079 - i) * 2048 + j) * 3 + 1];
						frame.data[(i * 1920 + j) * 3 + 2] = frim[((1079 - i) * 2048 + j) * 3 + 0];
					}
				}
				writer << frame;
			}
			else {
				goto st;
			}
		}
	}
	else {
		while (GetMessage(&msg, nullptr, 0, 0))
		{
			if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			if (start) {
				goto st;
			}
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

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_QUEVO));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_QUEVO);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

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
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//


template <class T>
int solve3(T *m, T *y, T *x, int n, T *workspace) {
	int i;
	workspace[0] = m[1];
	for (i = 1; i < n; i++) {
		workspace[i*2+1]= m[i * 3] / workspace[(i - 1)*2];
		workspace[i*2] = m[i * 3 + 1] - workspace[i*2+1]*m[i * 3 - 1];
	}
	x[0] = y[0];
	for (i = 1; i < n; i++) {
		x[i] = y[i] - workspace[i*2+1]*x[i - 1];
	}
	x[n - 1] = x[n - 1] / workspace[(n - 1)*2];
	for (i = n - 2; i >= 0; i--) {
		x[i] = (x[i] - m[i * 3 + 2] * x[i + 1]) / workspace[i*2];
	}
	return 0;
}

int evo(double2 *v1, double2 *v2, double2 *ws, double dt, double dx, int size) {
	int i;
	for (i = 1; i < size - 1; i++) {
		ws[i] = v1[i] - double2{ 0,dt / 2 }*((2.0*v1[i] - v1[i - 1] - v1[i + 1])*(0.5 / dx / dx) + po[i] * v1[i]);
		ws[size + i * 3 + 0] = double2{ 0,dt / 2 }*(-0.5 / dx / dx);
		ws[size + i * 3 + 1] = double2{ 1,0 }+double2{ 0,dt / 2 }*(1.0 / dx / dx + po[i]);
		ws[size + i * 3 + 2] = double2{ 0,dt / 2 }*(-0.5 / dx / dx);
	}
	ws[1] = ws[1] - ws[size * 3 + 1 * 3 + 0] * v2[0];
	ws[size - 2] = ws[size - 2] - ws[size * 3 + (size - 2) * 3 + 2] * v2[size - 1];
	solve3(ws + size + 3, ws + 1, v2 + 1, size - 2, ws + size * 4);
	return 0;
}

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
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
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
			SwapBuffers(hdc1);
            EndPaint(hWnd, &ps);
        }
        break;
	case WM_CREATE: {
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
		glEnableClientState(GL_NORMAL_ARRAY);
		((bool(_stdcall*)(int))wglGetProcAddress("wglSwapIntervalEXT"))(1);

		glEnable(GL_TEXTURE_2D);
		glEnable(GL_RENDERBUFFER);
		glEnable(GL_FRAMEBUFFER);

		versha = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(versha, 1, &shaderv, NULL);
		glCompileShader(versha);
		frasha = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(frasha, 1, &shaderf, NULL);
		glCompileShader(frasha);
		pro = glCreateProgram();
		glAttachShader(pro, versha);
		glAttachShader(pro, frasha);
		glLinkProgram(pro);
		glUseProgram(pro);

		glGenFramebuffersEXT(1, &framebuffer);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);

		glGenRenderbuffersEXT(1, &depthbuffer);
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthbuffer);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, 2048, 2048);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthbuffer);

		glGenTextures(1, &texbuffer);
		glBindTexture(GL_TEXTURE_2D, texbuffer);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2048, 2048, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texbuffer, 0);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

		ang1 = PI * 0.5;
		ang2 = 0;
		len = 4.5;
		cenx = 0.0;
		ceny = 0.0;
		cenz = 0.0;
		for (int i = 0; i < size; i++) {
			double x = l + (i + 0.5)*((r - l) / size);
			v1[i] = 2.0*exp(-0.0625/16 * (x - 32) * (x - 32)) *double2 { sin(18.76 * x), cos(18.76 * x) };
			//v1[i] = exp(-1*(x-6.0) * (x-6.0)) *double2 { 1, 0 };
			v2[i] = v1[i];
			po[i] = 0;
			if (-1.6 < x&&x < -1.5) {
				po[i] = 500.0;
			}
			float a = x- floor(x );
			float t = x - floor(x)-0.5;
			float b = ((7.0 / 16.0 < a) && (a < 9.0 / 16.0)) ? 1.0 : 0.0;
			po[i] = (b - 320.0 / 2000)*(x*x <36 ? 1.0 : 0.0) * 125.0;
			//po[i] = (x1<0.0625&&x1>-0.0625&&x > -6 && x < 6) ? 125 : 0;
			po[i] = (-4 < x&&x < 4) ? ((t<0.0625&&t>-0.0625) ? 105 : -20) : 0;
			//po[i] = 2.0*x*x - 72;
			//po[i] = (x<0.2&&x>-0.2) ? 125.0 : 0.0;
			//po[i] = (x<0.5&&x>-0.5) ? abs(x) * 200 : 0.0;
			popoint[i] = { x,po[i] * (1.0 / 1024.0),0 };

		}
		gettri(v1, point,pointn,l, r, size, seq, ri);
		break;
	}
	case WM_SIZE: {
		cx = lParam & 0xffff;
		cy = (lParam & 0xffff0000) >> 16;
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-(float)cx / (cx + cy) *len*0.0078125, (float)cx / (cx + cy) *len*0.0078125, -(float)cy / (cx + cy)*len*0.0078125, (float)cy / (cx + cy) *len*0.0078125, len*0.00390625, len*100.0);
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
			l = len * 5.0 / (cx + cy);
			cenx += l * (-(x - mx)*sin(ang1) - (y - my)*sin(ang2)*cos(ang1));
			//ceny += l * ((y - my)* cos(ang2));
			//cenz += l * ((x - mx)*cos(ang1) - (y - my)*sin(ang2)*sin(ang1));
		}
		mx = x;
		my = y;
		if (f) {
			draw();
			SwapBuffers(hdc1);
		}
		break;
	}
	case WM_MOUSEWHEEL: {
		short m;
		m = (wParam & 0xffff0000) >> 16;
		len *= exp(-m * 0.001);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-(float)cx / (cx + cy) *len*0.0078125, (float)cx / (cx + cy) *len*0.0078125, -(float)cy / (cx + cy)*len*0.0078125, (float)cy / (cx + cy) *len*0.0078125, len*0.00390625, len*100.0);
		draw();
		SwapBuffers(hdc1);
		break;
	}
	case WM_KEYDOWN: {
		switch (wParam) {
		case(32):
			start = !start;
			if (start) {

				int index = fileindex();
				char info[1024];
				sprintf_s(info, "D:/files/courses/qu/img/%d.mp4", index);
				writer.open(info, cv::VideoWriter::fourcc('H', '2', '6', '4'), 60.0, cv::Size(1920, 1080)); glViewport(0, 0, 1920, 1080);

				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glFrustum(-(float)1920 / (1920 + 1080) *len*0.0078125, (float)1920 / (1920 + 1080) *len*0.0078125, -(float)1080 / (1920 + 1080)*len*0.0078125, (float)1080 / (1920 + 1080) *len*0.0078125, len*0.00390625, len*100.0);
				draw();
				glFinish();

				glViewport(0, 0, cx, cy);
				glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glFrustum(-(float)cx / (cx + cy) *len*0.0078125, (float)cx / (cx + cy) *len*0.0078125, -(float)cy / (cx + cy)*len*0.0078125, (float)cy / (cx + cy) *len*0.0078125, len*0.00390625, len*100.0);

				draw();
				SwapBuffers(hdc1);

				glBindTexture(GL_TEXTURE_2D, texbuffer);
				glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, frim);
				for (int i = 0; i < 1080; i++) {
					for (int j = 0; j < 1920; j++) {
						frame.data[(i * 1920 + j) * 3 + 0] = frim[((1079 - i) * 2048 + j) * 3 + 2];
						frame.data[(i * 1920 + j) * 3 + 1] = frim[((1079 - i) * 2048 + j) * 3 + 1];
						frame.data[(i * 1920 + j) * 3 + 2] = frim[((1079 - i) * 2048 + j) * 3 + 0];
					}
				}
				writer << frame;
			}
			else {
				writer.release();
			}
			break;
		}
		break;
	}

    case WM_DESTROY:
        PostQuitMessage(0);
        break;
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
